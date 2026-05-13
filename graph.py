from __future__ import annotations

import asyncio
import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional

from langgraph.graph import END, StateGraph

from hotel_booking_agent.hotel_service import search_hotels, format_hotels_response

from .hotel_budget import filter_hotel_response_by_budget
from .providers import search_flights, search_trains
from .schemas import Budget, HotelOption, Itinerary, TravelOption, TravelState
from .train_price import enrich_train_prices


ORCHESTRATOR_PROMPT = (
    "Would you like a travel plan? Can I proceed with train or flight booking "
    "and later hotel booking?"
)

TRIP_KEYWORDS = {
    "plan", "trip", "itinerary", "travel", "visit", "tour", "vacation",
    "holiday", "flight", "train", "booking", "book", "hotel", "stay",
    "days in", "get to", "go to", "reach"
}
ASSISTANT_TRIP_KEYWORDS = {
    "itinerary", "travel plan", "day-wise", "flight", "train", "hotel booking",
    "book hotels", "trip plan"
}

REQUIRED_FIELDS = ["source", "destination", "budget", "travel_date", "duration"]


def is_trip_related(user_input: str, assistant_reply: str = "") -> bool:
    user_text = user_input.lower()
    assistant_text = assistant_reply.lower()
    return (
        any(keyword in user_text for keyword in TRIP_KEYWORDS)
        or any(keyword in assistant_text for keyword in ASSISTANT_TRIP_KEYWORDS)
    )


def _text_from_history(history: List[Dict[str, Any]]) -> str:
    return " ".join(
        str(item.get("content", ""))
        for item in history[-8:]
        if item.get("role") == "user"
    )


def _extract_budget(text: str) -> Optional[float]:
    patterns = [
        r"(?:budget|under|below|less than|rs\.?|inr)\s*(?:is|of|:)?\s*([0-9][0-9,]*)",
        r"([0-9][0-9,]*)\s*(?:rupees?|rs\.?|inr)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(",", ""))
    return None


def _extract_specific_budget(text: str, labels: Iterable[str]) -> Optional[float]:
    label_pattern = "|".join(re.escape(label) for label in labels)
    patterns = [
        rf"\b(?:{label_pattern})\s+budget\s*(?:is|of|:)?\s*(?:rs\.?|inr|₹)?\s*([0-9][0-9,]*)",
        rf"\b(?:rs\.?|inr|₹)\s*([0-9][0-9,]*)\s*(?:for\s+)?(?:{label_pattern})\b",
        rf"\b([0-9][0-9,]*)\s*(?:rupees?|rs\.?|inr)\s*(?:for\s+)?(?:{label_pattern})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(",", ""))
    return None


def _extract_total_budget(text: str) -> Optional[float]:
    explicit_patterns = [
        r"\b(?:total|overall|trip|total\s+trip)\s+budget\s*(?:is|of|:)?\s*(?:rs\.?|inr|₹)?\s*([0-9][0-9,]*)",
        r"\b(?:rs\.?|inr|₹)\s*([0-9][0-9,]*)\s*(?:total|overall|trip)?\s*budget\b",
        r"\b([0-9][0-9,]*)\s*(?:rupees?|rs\.?|inr)\s*(?:total|overall|trip)?\s*budget\b",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(",", ""))

    generic_match = re.search(
        r"\bbudget\s*(?:is|of|:)?\s*(?:rs\.?|inr|₹)?\s*([0-9][0-9,]*)",
        text,
        re.IGNORECASE,
    )
    if generic_match:
        prefix = text[max(0, generic_match.start() - 16):generic_match.start()].lower()
        if not re.search(r"\b(?:flight|train|travel|hotel)\s*$", prefix):
            return float(generic_match.group(1).replace(",", ""))

    return _extract_budget(text)


def _extract_budget_details(text: str) -> Dict[str, Optional[float]]:
    travel_budget = _extract_specific_budget(text, ("flight", "train", "travel"))
    hotel_budget = _extract_specific_budget(text, ("hotel",))
    total_budget = _extract_total_budget(text)
    if total_budget in {travel_budget, hotel_budget} and not re.search(
        r"\b(?:total|overall|trip|total\s+trip)\s+budget\b",
        text,
        re.IGNORECASE,
    ):
        total_budget = None
    return {
        "total_budget": total_budget,
        "travel_budget": travel_budget,
        "hotel_budget": hotel_budget,
    }


def _extract_travelers(text: str) -> Optional[int]:
    match = re.search(r"(\d+)\s*(?:people|persons|travellers|travelers|adults|passengers)", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _extract_duration(text: str) -> Optional[int]:
    match = re.search(r"(\d+)\s*(?:day|days|night|nights)", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _extract_date(text: str) -> Optional[str]:
    lowered = text.lower()
    if re.search(r"\btoday\b", lowered):
        return date.today().isoformat()
    if re.search(r"\btomorrow\b", lowered):
        return (date.today() + timedelta(days=1)).isoformat()

    patterns = [
        r"\b(\d{4}-\d{2}-\d{2})\b",
        r"\b(\d{1,2}/\d{1,2}/\d{4})\b",
        r"\b(\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+,?\s+\d{4})\b",
        r"\b([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def _parse_travel_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    cleaned = re.sub(r"(\d{1,2})(st|nd|rd|th)\b", r"\1", value.strip(), flags=re.IGNORECASE)
    cleaned = cleaned.replace(",", "")
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y"):
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


def _extract_destination(text: str, known_destinations: Iterable[str]) -> Optional[str]:
    lowered = text.lower()
    explicit_match = re.search(
        r"\bdestination(?:\s+city)?\s*(?:is|:)?\s*([A-Za-z][A-Za-z\s]{1,30}?)(?:\s*,|\s+budget|\s+date|$)",
        text,
        re.IGNORECASE,
    )
    if explicit_match:
        return explicit_match.group(1).strip().title()

    ordered = sorted({d.strip() for d in known_destinations if d.strip()}, key=len, reverse=True)
    for destination in ordered:
        if destination.lower() in lowered:
            return destination.title()
    to_match = re.search(r"\b(?:to|visit|in)\s+([A-Za-z][A-Za-z\s]{2,30})", text, re.IGNORECASE)
    if to_match:
        return to_match.group(1).strip().title()
    return None


def _extract_route(text: str, known_destinations: Iterable[str]) -> tuple[Optional[str], Optional[str]]:
    route_match = re.search(
        r"\bfrom\s+([A-Za-z][A-Za-z\s]{2,30}?)\s+to\s+([A-Za-z][A-Za-z\s]{2,30}?)(?:\s+(?:budget|date|on|for|under|inr|rs\.?|\d)|$)",
        text,
        re.IGNORECASE,
    )
    if not route_match:
        return None, None

    raw_source = route_match.group(1).strip()
    raw_destination = route_match.group(2).strip()
    known = {destination.lower(): destination.title() for destination in known_destinations if destination.strip()}

    source = known.get(raw_source.lower(), raw_source.title())
    destination = known.get(raw_destination.lower(), raw_destination.title())
    return source, destination


def _extract_source(text: str, destination: Optional[str]) -> Optional[str]:
    patterns = [
        r"\bfrom\s+([A-Za-z][A-Za-z\s]{2,30})\s+(?:to|towards|for)\b",
        r"\bsource(?:\s+city)?\s*(?:is|:)?\s*([A-Za-z][A-Za-z\s]{1,30}?)(?:\s+and\s+travel\s+date|\s+travel\s+date|\s+destination|\s+budget|\s+date|\s*,|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            source = match.group(1).strip().title()
            if not destination or source.lower() != destination.lower():
                return source
    return None


def _state_from_dict(raw: Dict[str, Any]) -> TravelState:
    return TravelState.model_validate(raw)


def input_parser_node(raw_state: Dict[str, Any]) -> Dict[str, Any]:
    state = _state_from_dict(raw_state)
    combined_text = f"{_text_from_history(state.phase1_history)} {state.user_input}"

    current_route_source, current_route_destination = _extract_route(state.user_input, state.known_destinations)
    current_destination = current_route_destination or _extract_destination(state.user_input, state.known_destinations)
    current_source = current_route_source or _extract_source(state.user_input, current_destination)
    has_explicit_destination = bool(
        current_route_destination
        or re.search(r"\bdestination(?:\s+city)?\s*(?:is|:)?", state.user_input, re.IGNORECASE)
    )

    history_route_source, history_route_destination = _extract_route(combined_text, state.known_destinations)
    destination = current_destination or history_route_destination or _extract_destination(
        combined_text,
        state.known_destinations,
    )
    source = current_source or history_route_source or state.trip_details.source

    latest_city = current_destination
    if latest_city and not current_source and not has_explicit_destination and state.trip_details.destination and not state.trip_details.source:
        if latest_city.lower() != state.trip_details.destination.lower():
            source = latest_city
            destination = state.trip_details.destination
    elif latest_city and not current_source and not has_explicit_destination and state.trip_details.source and not state.trip_details.destination:
        if latest_city.lower() != state.trip_details.source.lower():
            destination = latest_city
    current_budgets = _extract_budget_details(state.user_input)
    history_budgets = _extract_budget_details(combined_text)
    total_budget = current_budgets["total_budget"]
    if total_budget is None:
        total_budget = history_budgets["total_budget"]
    travel_budget = current_budgets["travel_budget"]
    if travel_budget is None:
        travel_budget = history_budgets["travel_budget"]
    hotel_budget = current_budgets["hotel_budget"]
    if hotel_budget is None:
        hotel_budget = history_budgets["hotel_budget"]
    travelers = _extract_travelers(state.user_input) or _extract_travelers(combined_text)
    duration = _extract_duration(state.user_input) or _extract_duration(combined_text)
    start_date = _extract_date(state.user_input) or _extract_date(combined_text)

    if source:
        state.trip_details.source = source
    if destination:
        state.trip_details.destination = destination
    if total_budget is not None:
        state.budget.total_budget = total_budget
    if travel_budget is not None:
        state.budget.travel_budget = travel_budget
    if hotel_budget is not None:
        state.budget.hotel_budget = hotel_budget
    if state.budget.total_budget is None and state.budget.travel_budget is not None and state.budget.hotel_budget is not None:
        state.budget.total_budget = state.budget.travel_budget + state.budget.hotel_budget
    if travelers:
        state.trip_details.travelers = travelers
    if duration:
        state.trip_details.duration_days = duration
    if start_date:
        state.trip_details.start_date = start_date

    state.metadata["history_scope"] = "current_user_current_session_blob"
    state.status = "PARSED"
    return state.model_dump()


def missing_info_node(raw_state: Dict[str, Any]) -> Dict[str, Any]:
    state = _state_from_dict(raw_state)
    missing = []
    if not state.trip_details.source:
        missing.append("source")
    if not state.trip_details.destination:
        missing.append("destination")
    if not state.budget.total_budget:
        missing.append("budget")
    if not state.trip_details.start_date:
        missing.append("travel_date")
    if not state.trip_details.duration_days:
        missing.append("duration")
    state.missing_fields = missing
    if missing:
        label_map = {
            "source": "source city",
            "destination": "destination city",
            "budget": "total trip budget",
            "travel_date": "travel date",
            "duration": "number of days for the trip",
        }
        label = " and ".join(label_map.get(field, field) for field in missing)
        state.messages.append(f"I can start the travel plan, but I need your {label} first.")
        state.status = "NEEDS_INFO"
    else:
        state.status = "READY_FOR_DATE_WINDOW"
    return state.model_dump()


def date_window_node(raw_state: Dict[str, Any]) -> Dict[str, Any]:
    state = _state_from_dict(raw_state)
    today = date.today()
    travel_date = _parse_travel_date(state.trip_details.start_date)

    state.metadata["today"] = today.isoformat()
    if travel_date is None:
        state.missing_fields = ["travel_date"]
        state.messages.append("I can start the travel plan, but I need a valid travel date first.")
        state.status = "NEEDS_INFO"
        return state.model_dump()

    normalized_date = travel_date.isoformat()
    state.trip_details.start_date = normalized_date
    state.metadata["requested_travel_date"] = normalized_date

    if travel_date < today:
        state.filtered_options = []
        state.travel_options = []
        state.errors.append("Past travel dates are not possible for train or flight search.")
        state.messages.append("Sorry, that is not possible because the travel date is in the past.")
        state.status = "PAST_DATE_NOT_POSSIBLE"
        return state.model_dump()

    if travel_date == today:
        search_dates = [today + timedelta(days=offset) for offset in range(0, 4)]
    else:
        search_dates = []
        for offset in range(-3, 4):
            candidate = travel_date + timedelta(days=offset)
            if candidate >= today:
                search_dates.append(candidate)

    state.metadata["travel_search_dates"] = [item.isoformat() for item in search_dates]
    state.metadata["travel_date_window_rule"] = (
        "today_plus_3_days" if travel_date == today else "future_date_minus_3_plus_3_days"
    )
    state.status = "DATE_WINDOW_READY"
    return state.model_dump()


def budget_node(raw_state: Dict[str, Any]) -> Dict[str, Any]:
    state = _state_from_dict(raw_state)
    total = state.budget.total_budget
    explicit_travel_budget = state.budget.travel_budget
    explicit_hotel_budget = state.budget.hotel_budget
    if total:
        travel_budget = explicit_travel_budget if explicit_travel_budget is not None else round(total * 0.45, 2)
        hotel_budget = explicit_hotel_budget if explicit_hotel_budget is not None else round(total * 0.55, 2)
        state.budget = Budget(
            total_budget=total,
            travel_budget=travel_budget,
            hotel_budget=hotel_budget,
            flexible=True,
        )
        state.metadata["budget_split_rule"] = "default_45_travel_55_hotel"
        state.metadata["explicit_travel_budget"] = explicit_travel_budget is not None
        state.metadata["explicit_hotel_budget"] = explicit_hotel_budget is not None
        state.metadata["budget_summary"] = (
            f"Total budget: INR {int(total)}, Travel budget: INR {int(travel_budget)}, "
            f"Hotel budget: INR {int(hotel_budget)}"
        )
    else:
        state.budget = Budget(flexible=True)
    state.status = "BUDGET_READY"
    return state.model_dump()


def travel_search_node(raw_state: Dict[str, Any]) -> Dict[str, Any]:
    state = _state_from_dict(raw_state)

    async def run_parallel() -> tuple[List[TravelOption], List[str], Dict[str, int]]:
        results = await asyncio.gather(
            search_flights(state),
            search_trains(state),
            return_exceptions=True,
        )
        options: List[TravelOption] = []
        errors: List[str] = []
        counts: Dict[str, int] = {}
        labels = ["flight", "train"]
        for label, result in zip(labels, results):
            if isinstance(result, Exception):
                counts[label] = 0
                message = str(result)
                if message in {"rate limit exceed for flight agent", "rate limit exceed for train agent"}:
                    errors.append(message)
                else:
                    errors.append(f"{label.title()} agent failed: {message}")
            else:
                counts[label] = len(result)
                if not result:
                    errors.append(f"{label.title()} agent returned 0 option(s).")
                options.extend(result)
        return options, errors, counts

    options, errors, counts = asyncio.run(run_parallel())
    state.travel_options = options
    state.errors.extend(errors)
    state.metadata["agent_option_counts"] = counts
    state.metadata["parallel_agents_completed"] = ["flight", "train"]
    enriched_options, train_price_data, train_price_error = enrich_train_prices(state)
    state.travel_options = enriched_options
    if train_price_data:
        state.metadata["train_price_data"] = train_price_data
    if train_price_error:
        state.errors.append(train_price_error)
    state.status = "TRAVEL_AGENTS_JOINED"
    return state.model_dump()


def aggregator_node(raw_state: Dict[str, Any]) -> Dict[str, Any]:
    state = _state_from_dict(raw_state)
    
    unique_options = {}
    for option in state.travel_options:
        key = (option.name, option.departure_time)
        if key not in unique_options or option.price < unique_options[key].price:
            unique_options[key] = option
            
    state.travel_options = sorted(
        unique_options.values(),
        key=lambda option: (not option.available, option.price, option.duration_hours),
    )
    state.status = "TRAVEL_OPTIONS_AGGREGATED"
    return state.model_dump()


def _top_options_by_mode(options: List[TravelOption], limit_per_mode: int = 5) -> List[TravelOption]:
    selected: List[TravelOption] = []
    for mode in ("flight", "train"):
        selected.extend([option for option in options if option.mode == mode][:limit_per_mode])
    return selected


def budget_filter_node(raw_state: Dict[str, Any]) -> Dict[str, Any]:
    state = _state_from_dict(raw_state)
    travel_budget = state.budget.travel_budget
    if travel_budget:
        filtered: List[TravelOption] = []
        for mode in ("flight", "train"):
            mode_options = [option for option in state.travel_options if option.mode == mode]
            if mode == "train" and any(option.price == 0 for option in mode_options):
                filtered.extend(mode_options[:5])
                if mode_options:
                    state.errors.append("Train prices unavailable/live search-only, showing train options without fare filtering.")
                continue
            within_budget = [option for option in mode_options if option.price <= travel_budget]
            if within_budget:
                filtered.extend(within_budget[:5])
            else:
                filtered.extend(mode_options[:5])
                if mode_options:
                    if mode == "flight":
                        state.errors.append(
                            f"No flight option fit INR {int(travel_budget)}, showing closest flight options."
                        )
                    else:
                        state.errors.append(
                            f"No {mode} option fit the travel budget exactly, so I kept the closest {mode} options."
                        )
        state.filtered_options = filtered
    else:
        state.filtered_options = _top_options_by_mode(state.travel_options)
    state.metadata["filtered_option_counts"] = {
        "flight": len([option for option in state.filtered_options if option.mode == "flight"]),
        "train": len([option for option in state.filtered_options if option.mode == "train"]),
    }
    state.status = "AWAITING_TRAVEL_SELECTION"
    return state.model_dump()


def show_options_node(raw_state: Dict[str, Any]) -> Dict[str, Any]:
    state = _state_from_dict(raw_state)
    budget_summary = state.metadata.get("budget_summary")
    if budget_summary and budget_summary not in state.messages:
        state.messages.append(str(budget_summary))
    for error in state.errors:
        if error not in state.messages:
            state.messages.append(error)
    count = len(state.filtered_options)
    if count:
        counts = state.metadata.get("filtered_option_counts") or {}
        flight_count = counts.get("flight", len([option for option in state.filtered_options if option.mode == "flight"]))
        train_count = counts.get("train", len([option for option in state.filtered_options if option.mode == "train"]))
        state.messages.append(
            f"I found {flight_count} flight option(s) and {train_count} train option(s). "
            "Choose one to continue to hotel booking."
        )
    else:
        state.messages.append("I could not find train or flight options yet. Share source and destination to retry.")
    return state.model_dump()


def _next_after_missing(raw_state: Dict[str, Any]) -> str:
    state = _state_from_dict(raw_state)
    return "end" if state.missing_fields else "date_window"


def _next_after_date_window(raw_state: Dict[str, Any]) -> str:
    state = _state_from_dict(raw_state)
    return "end" if state.status in {"PAST_DATE_NOT_POSSIBLE", "NEEDS_INFO"} else "budget"


def build_orchestrator_graph():
    graph = StateGraph(dict)
    graph.add_node("input_parser", input_parser_node)
    graph.add_node("missing_info", missing_info_node)
    graph.add_node("date_window", date_window_node)
    graph.add_node("budget", budget_node)
    graph.add_node("travel_search", travel_search_node)
    graph.add_node("aggregator", aggregator_node)
    graph.add_node("budget_filter", budget_filter_node)
    graph.add_node("show_options", show_options_node)

    graph.set_entry_point("input_parser")
    graph.add_edge("input_parser", "missing_info")
    graph.add_conditional_edges(
        "missing_info",
        _next_after_missing,
        {"date_window": "date_window", "end": END},
    )
    graph.add_conditional_edges(
        "date_window",
        _next_after_date_window,
        {"budget": "budget", "end": END},
    )
    graph.add_edge("budget", "travel_search")
    graph.add_edge("travel_search", "aggregator")
    graph.add_edge("aggregator", "budget_filter")
    graph.add_edge("budget_filter", "show_options")
    graph.add_edge("show_options", END)
    return graph.compile()


GRAPH = build_orchestrator_graph()


def start_orchestrator(
    user_input: str,
    phase1_history: List[Dict[str, Any]],
    known_destinations: List[str],
    session_id: str,
    previous_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    seed = previous_state or {}
    state = TravelState.model_validate(
        {
            **seed,
            "user_input": user_input,
            "phase1_history": phase1_history[-8:],
            "known_destinations": known_destinations,
            "session_id": session_id,
            "history_source": "azure_blob_storage",
        }
    )
    final_state = GRAPH.invoke(
        state.model_dump(),
        config={"run_name": "travel_orchestrator", "tags": ["orchestrator", "backend"]},
    )
    return TravelState.model_validate(final_state).model_dump()


def dismiss_orchestrator(session: Dict[str, Any]) -> Dict[str, Any]:
    orchestrator = session.setdefault("orchestrator", {})
    orchestrator["active"] = False
    orchestrator["awaiting_activation"] = False
    orchestrator["last_prompt_message_id"] = None
    return session


def _format_option(option: TravelOption) -> str:
    hours = int(option.duration_hours)
    minutes = int(round((option.duration_hours - hours) * 60))
    duration_str = f"{hours}h {minutes}m" if minutes > 0 else f"{hours}h"
    
    return (
        f"{option.name} ({option.mode}) - {option.source} to {option.destination}, "
        f"{duration_str}, INR {int(option.price)}"
    )


def continue_after_travel_selection(
    state_data: Dict[str, Any],
    selected_option_id: str,
    phase1_history: List[Dict[str, Any]],
    known_destinations: List[str],
    session_id: str,
) -> Dict[str, Any]:
    state = TravelState.model_validate(state_data)
    options = state.filtered_options or state.travel_options
    selected = next((option for option in options if option.id == selected_option_id), None)
    if selected is None:
        state.errors.append("Selected travel option was not found.")
        state.status = "AWAITING_TRAVEL_SELECTION"
        return state.model_dump()

    state.selected_option = selected
    state.status = "HOTEL_SEARCHING"
    
    try:
        check_in_obj = datetime.strptime(state.trip_details.start_date, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        check_in_obj = date.today() + timedelta(days=1)
        
    duration = state.trip_details.duration_days or 1
    check_out_obj = check_in_obj + timedelta(days=duration)
    
    payload = {
        "destination_city": selected.destination,
        "check_in_date": check_in_obj.strftime("%Y-%m-%d"),
        "check_out_date": check_out_obj.strftime("%Y-%m-%d"),
        "adults": state.trip_details.travelers or 2,
        "children": 0
    }
    
    hotels = []
    try:
        hotels = search_hotels(payload)
    except Exception as e:
        state.errors.append(f"Hotel search failed: {e}")

    state.hotel_options = [
        HotelOption(
            name=str(item.get("hotel_name", "Hotel option")),
            destination=item.get("city"),
            price_per_night=item.get("price_per_night"),
            rating=item.get("rating"),
            source="serpapi",
            raw=item,
        )
        for item in hotels
    ]
    state.selected_hotel = state.hotel_options[0] if state.hotel_options else None

    # Format the message for the user:
    if hotels:
        raw_hotels_message = format_hotels_response(hotels, payload)
        hotels_message = filter_hotel_response_by_budget(
            raw_hotels_message,
            state.budget.hotel_budget,
            duration,
        )
    else:
        hotels_message = f"I could not find any live hotels for {selected.destination} matching your dates."

    hotel_name = state.selected_hotel.name if state.selected_hotel else "a hotel option"
    state.itinerary = Itinerary(
        travel=selected,
        hotel=state.selected_hotel,
        summary=(
            f"Travel option selected: {_format_option(selected)}. "
            f"Hotel step completed with {hotel_name}."
        ),
    )
    
    state.hotel_payload = {"answer": f"Great. I will use that travel option and now check hotels.\n\n{hotels_message}"}
    
    state.messages.append("Great. I will use that travel option and now check hotels.")
    state.messages.append(hotels_message)
    state.messages.append(state.itinerary.summary)
    
    state.status = "FINAL_PLAN_READY"
    return state.model_dump()
