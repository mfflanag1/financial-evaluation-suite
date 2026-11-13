"""Independent finite decision audits of five fixed public core contracts.

Only public question packets supply financial inputs. Constants transcribed
from prose are guarded by the canonical public-question hashes below; changing
a contract requires a new review. Helpers expose the arithmetic for boundary
tests, not a general execution service. No scorer or stored answer is imported.
"""
from fractions import Fraction as F
import hashlib
import itertools
import json
import re


REVIEWED_QUESTIONS = {
    "c205": "fa832c97906e5e9145d17728789d3bdadf7168ab981e32eae590d2e8a153df5d",
    "c206": "ac84b76e879dc4d8794c77d7b7582a5b96c11e23e225c977c411a85be1df6653",
    "c213": "720059874cbc781b755f446feee2e289d946355498e7b0cb49b21453e3fa6ccb",
    "c217": "91a26d3513e89f6ff158a8b849213d9176cd7529edc02e3d0db03c08b2ec40eb",
    "c218": "1003eb23f50e85f109230b587448f79d661ff1bd432c6315d811e01a7dc76abb",
}


def scope(question, case_id):
    encoded = json.dumps(question, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    actual = hashlib.sha256(encoded).hexdigest()
    if actual != REVIEWED_QUESTIONS[case_id]:
        raise ValueError("Public decision contract differs from reviewed input")
    return {"canonical_question_sha256": actual,
            "public_contract_ids": [source["id"] for source in question["sources"]]}


def research_policies(cells, payoffs, costs, budget, distinct=True):
    """Enumerate every allowed deterministic two-query policy with exact values.

    cells are (probability, state, {query: bit}); policy branches always use
    first-bit then second-bit order 0,1. Unpurchased bits never affect actions.
    """
    actions = sorted(payoffs)
    policies = []

    def evaluate(first, seconds, terminal):
        gross = fees = queried_mass = second_positive_mass = F(0)
        for probability, state, bits in cells:
            branch = 0 if first == "none" else bits[first]
            second = seconds[branch]
            bit = 0 if second == "stop" else bits[second]
            gross += probability * payoffs[terminal[2 * branch + bit]][state]
            if first != "none":
                fees += probability * costs[first]
            if second != "stop":
                fees += probability * costs[second]
                queried_mass += probability
                second_positive_mass += probability * bits[second]
        return {"policy": (first, tuple(seconds), tuple(terminal)), "gross": gross,
                "fees": fees, "net": gross - fees, "queried_mass": queried_mass,
                "second_positive": second_positive_mass / queried_mass if queried_mass else F(0)}

    for action in actions:
        policies.append(evaluate("none", ("stop", "stop"), (action,) * 4))
    for first in sorted(costs):
        if costs[first] > budget:
            continue
        continuations = [("stop", (action, action)) for action in actions]
        for second in sorted(costs):
            if (distinct and second == first) or costs[first] + costs[second] > budget:
                continue
            continuations.extend((second, pair) for pair in itertools.product(actions, repeat=2))
        for left, right in itertools.product(continuations, repeat=2):
            policies.append(evaluate(first, (left[0], right[0]), left[1] + right[1]))
    policies.sort(key=lambda row: (-row["net"], row["policy"]))
    return policies


def policy_certificate(rows):
    best = rows[0]
    optima = [row["policy"] for row in rows if row["net"] == best["net"]]
    return {"enumerated_policies": len(rows), "all_optimal_policies": optima,
            "optimal_net_fraction": str(best["net"]),
            "certificate_branch_order": "First bit 0,1; within each branch second bit 0,1.",
            "tie_policy": "Exact value, then ordinary lexicographic policy tuple."}


def c205(question):
    certificate = scope(question, "c205")
    inputs = question["inputs"]
    probabilities = [F(str(value)) for value in inputs["joint_state_probabilities"]]
    bits = ((1, 1), (1, 0), (0, 1), (0, 0))
    cells = [(probability, index, dict(zip(("Q", "R"), signal)))
             for index, (probability, signal) in enumerate(zip(probabilities, bits))]
    rows = research_policies(cells, inputs["project_payoffs"],
                             {name: F(inputs["query_cost"]) for name in ("Q", "R")},
                             2 * inputs["query_cost"], distinct=False)
    best = rows[0]
    first, seconds, terminal = best["policy"]
    no_query = max(row["net"] for row in rows if row["policy"][0] == "none")
    certificate.update(policy_certificate(rows))
    certificate["query_repeat_scope"] = "Repeated tests are enumerated and reveal the original fixed bit."
    certificate["contract_tie_scope"] = "Every global optimum is allowed; lexicographic order selects a reproducible audit representative."
    return {"first_query": first,
            "query_again_after": [label for bit, label in ((1, "positive"), (0, "negative"))
                                  if seconds[bit] != "stop"],
            "terminal_actions": [terminal[index] for index in (3, 2, 1, 0)],
            "second_positive_probability": float(best["second_positive"]),
            "expected_query_cost": float(best["fees"]),
            "optimal_net_value": float(best["net"]), "no_information_value": float(no_query),
            "information_value": float(best["net"] - no_query)}, certificate


def replay_orders(opening_orders, events):
    """Reconcile exchange-ordered fills and accepted commitments.

    Cancellation is terminal even when an earlier fill is later corrected.
    Identical event/execution deliveries are idempotent; local requests are not
    exchange events and callers account for their explicit reservation rules.
    """
    orders = {row["id"]: dict(row, closed=False) for row in opening_orders}
    fills = {}
    seen = {}
    for event in sorted(events, key=lambda row: row["sequence"]):
        sequence = event["sequence"]
        if sequence in seen:
            if event != seen[sequence]:
                raise ValueError("Conflicting duplicate exchange event")
            continue
        seen[sequence] = event
        order = orders[event["order"]]
        kind = event["type"]
        if kind == "fill":
            fill = {key: event[key] for key in ("order", "quantity", "price", "day")}
            if event["fill_id"] in fills:
                if fills[event["fill_id"]] != fill:
                    raise ValueError("Conflicting duplicate execution")
            else:
                fills[event["fill_id"]] = fill
        elif kind == "bust":
            fill = fills[event["fill_id"]]
            if not 0 <= event["quantity"] <= fill["quantity"]:
                raise ValueError("Bust exceeds surviving execution")
            fill["quantity"] -= event["quantity"]
        elif kind == "correction":
            fill = fills[event["fill_id"]]
            fill.update({key: event[key] for key in ("quantity", "price", "day")})
        elif kind == "replace":
            order.update(total=event["total"], limit=event["limit"])
        elif kind == "cancel_ack":
            order["closed"] = True
        else:
            raise ValueError("Unsupported exchange event")
    for order in orders.values():
        filled = sum(fill["quantity"] for fill in fills.values() if fill["order"] == order["id"])
        order["remaining"] = 0 if order["closed"] else order["total"] - filled
        if order["remaining"] < 0:
            raise ValueError("Execution exceeds accepted total")
    return orders, fills


def c206(question):
    certificate = scope(question, "c206")
    inputs = question["inputs"]
    events = [dict(row, sequence=row["exchange_sequence"], day=0)
              for row in inputs["received_events_in_arrival_order"] if "exchange_sequence" in row]
    orders, fills = replay_orders(inputs["opening_accepted_orders"], events)
    inventory = inputs["opening_inventory"]
    cash = F(inputs["opening_cash"])
    for fill in fills.values():
        sign = 1 if orders[fill["order"]]["side"] == "buy" else -1
        inventory += sign * fill["quantity"]
        cash -= sign * fill["quantity"] * fill["price"]
    reserved_cash = sum(order["remaining"] * order["limit"] for order in orders.values() if order["side"] == "buy")
    reserved_shares = inputs["delivery_allocation_shares"]
    reserved_shares += sum(order["remaining"] for order in orders.values() if order["side"] == "sell")
    reserved_shares += sum(row["quantity"] for row in inputs["received_events_in_arrival_order"]
                           if row["type"] == "new_sell_request" and not row["acknowledged"])
    available = inventory - reserved_shares
    buy_price, sell_price = (inputs["hypothetical_new_prices"][side] for side in ("buy", "sell"))
    choices = []
    for sale in range(available + 1):
        # Fixed public mandate: 400 operating cash, 10-share purchase lots.
        maximum_lots = (cash + sale * sell_price - reserved_cash - 400) // (10 * buy_price)
        for lots in range(maximum_lots + 1):
            purchase = lots * 10
            choices.append((purchase, sale, cash + sale * sell_price - purchase * buy_price))
    purchase, sale, final_cash = min(choices, key=lambda row: (-row[0], row[1]))
    certificate.update({"surviving_fill_quantities": {key: fill["quantity"] for key, fill in fills.items()},
                        "accepted_remaining": {key: order["remaining"] for key, order in orders.items()},
                        "reserved_shares": reserved_shares, "feasible_actions": len(choices),
                        "cash_requirement": reserved_cash + 400,
                        "one_less_sale_cash": float(final_cash - sell_price),
                        "next_lot_at_maximum_sale_cash": float(cash + available * sell_price - (purchase + 10) * buy_price)})
    return {"available_inventory": available, "current_cash": float(cash), "current_inventory": inventory,
            "reserved_buy_cash": reserved_cash, "new_buy_quantity": purchase, "new_sell_quantity": sale,
            "final_cash": float(final_cash), "final_inventory": inventory - sale + purchase}, certificate


def repair_state(action):
    """Exact c213 ledger for a candidate within its published integer grid."""
    a, b, c, hedge, funding = action
    trades = [new - old for new, old in zip((a, b, c), (7, 6, 4))]
    buys = [max(0, trade) for trade in trades]
    sales = [max(0, -trade) for trade in trades]
    cost = sum(F(buy * buy, 2) + 2 * max(sale - 2, 0) + int(buy + sale > 0)
               for buy, sale in zip(buys, sales))
    cost += F(hedge + funding, 2) + int(hedge > 0) + int(funding > 0)
    early = 20 + 10 * sales[0] - 10 * sum(buys) - cost - 2 * hedge + 5 * funding - 40
    losses = (5 * a + 2 * b - c - 6 * hedge, -a + 5 * b + 3 * c - 2 * hedge)
    before_stress = early + 10 * sum(sales[1:]) + 2 * hedge - 5 * funding - 3
    final = tuple(before_stress - max(0, loss) for loss in losses)
    slacks = {"total_position": a + b + c - 10, "ab_limit": 11 - a - b,
              "ac_distance": 3 - abs(a - c), "turnover": 13 - sum(map(abs, trades)) - hedge,
              "borrowing_base_1": 2 * a + c - 5 * funding,
              "borrowing_base_2": a + 2 * c - 5 * funding,
              "early_cash": early, "scenario_1_cash": final[0], "scenario_2_cash": final[1]}
    objective = cost + F(3, 10) * max(losses) + F(1, 5) * (17 - a - b - c)
    return {"cost": cost, "early": early, "losses": losses, "final": final, "objective": objective,
            "slacks": slacks, "feasible": all(slack >= 0 for slack in slacks.values())}


def c213(question):
    certificate = scope(question, "c213")
    feasible = []
    for action in itertools.product(range(9), range(9), range(9), range(6), range(5)):
        state = repair_state(action)
        if state["feasible"]:
            feasible.append((state["objective"], action, state))
    feasible.sort(key=lambda row: row[:2])
    objective, action, state = feasible[0]
    certificate.update({"enumerated_actions": 9 ** 3 * 6 * 5, "feasible_actions": len(feasible),
                        "all_minimum_objective_actions": [row[1] for row in feasible if row[0] == objective],
                        "next_best_action": feasible[1][1], "objective_fraction": str(objective),
                        "next_best_objective_fraction": str(feasible[1][0]),
                        "selected_constraint_slacks": {key: float(value) for key, value in state["slacks"].items()}})
    return {"action": list(action), "early_cash": float(state["early"]), "implementation_cost": float(state["cost"]),
            "objective": float(objective), "next_best_objective": float(feasible[1][0]),
            "scenario_cash": list(map(float, state["final"])), "scenario_losses": list(state["losses"])}, certificate


def c217(question):
    certificate = scope(question, "c217")
    texts = {source["id"]: source["text"] for source in question["sources"]}
    weights = [list(map(int, block.split(","))) for block in re.findall(r"are\[([\d,]+)\]/100", texts["in_joint"])]
    if len(weights) != 2 or any(len(row) != 8 for row in weights) or sum(map(sum, weights)) != 100:
        raise ValueError("Expected the complete 16-cell joint law")
    cells = [(F(weight, 100), state, dict(zip("ABC", bits)))
             for state, row in enumerate(weights) for weight, bits in zip(row, itertools.product((0, 1), repeat=3))]
    payoffs = {"invest": (30, -40), "hedge": (10, -8), "pass": (0, 0)}
    rows = research_policies(cells, payoffs, {"A": F(1, 2), "B": F(1, 2), "C": F(3, 2)}, F(4))
    best = rows[0]
    first, seconds, terminal = best["policy"]
    no_query = max(row["net"] for row in rows if row["policy"][0] == "none")
    certificate.update(policy_certificate(rows))
    certificate["joint_probability_total"] = str(sum(cell[0] for cell in cells))
    return {"first_query": first, "second_queries": list(seconds), "terminal_actions": list(terminal),
            "expected_gross_payoff": float(best["gross"]), "expected_query_cost": float(best["fees"]),
            "optimal_net_value": float(best["net"]), "no_information_value": float(no_query),
            "incremental_information_value": float(best["net"] - no_query)}, certificate


def settlement_repair(action, cash=600, settled=70, old_cash=(0, -220),
                      old_shares=(2, 20), owned=92, reservation=20, reserve=(50, 100)):
    """c218 dated cash/share feasibility; defaults are reconciled public inputs."""
    sale, buy, bridge = action
    fee = F(bridge, 20)
    cash0 = cash + 9 * sale + bridge - 10 * buy
    cash1 = cash0 + old_cash[0] - 400
    cash2 = cash1 + old_cash[1] - bridge - fee
    inventory1 = settled - sale + old_shares[0] - 25
    final_owned = owned - sale + buy - 25
    slacks = {"sale_resource": settled - reservation - sale, "cash_T0": cash0,
              "cash_T1": cash1 - reserve[0], "cash_T2": cash2 - reserve[1],
              "inventory_T1": inventory1 - reservation, "inventory_T2": final_owned - reservation}
    in_grid = sale in range(0, 51, 5) and buy in range(0, 81, 5) and bridge in range(101)
    return {"fee": fee, "cash": (cash0, cash1, cash2), "inventory1": inventory1,
            "final_owned": final_owned, "slacks": slacks,
            "feasible": in_grid and all(slack >= 0 for slack in slacks.values())}


def c218(question):
    certificate = scope(question, "c218")
    texts = {source["id"]: source["text"] for source in question["sources"]}
    tape = texts["ex_tape"]
    events = []
    for sequence, order, fill, quantity, price, day in re.findall(
            r"(\d+) ([BS]) fill (F\d) quantity(\d+) at(\d+),? settles T(\d)", tape):
        events.append({"sequence": int(sequence), "order": order, "type": "fill", "fill_id": fill,
                       "quantity": int(quantity), "price": int(price), "day": int(day)})
    sequence, total, limit = map(int, re.search(r"(\d+) accepted B replacement sets TOTAL quantity(\d+) at(\d+)", tape).groups())
    events.append({"sequence": sequence, "order": "B", "type": "replace", "total": total, "limit": limit})
    sequence = int(re.search(r"(\d+) B cancel acknowledgement", tape).group(1))
    events.append({"sequence": sequence, "order": "B", "type": "cancel_ack"})
    sequence, fill, quantity, price, day = re.search(
        r"(\d+) correction of (F\d) replaces that entire execution with quantity(\d+) at(\d+), still settles T(\d)", tape).groups()
    events.append({"sequence": int(sequence), "order": "B", "type": "correction", "fill_id": fill,
                   "quantity": int(quantity), "price": int(price), "day": int(day)})
    # The duplicate message carries event 8's identity, not a new execution.
    events.append(dict(next(event for event in events if event["sequence"] == 8)))
    orders, fills = replay_orders([{"id": "B", "side": "buy", "total": 80, "limit": 10},
                                  {"id": "S", "side": "sell", "total": 30, "limit": 12}], events)
    cash = [0, 0]
    shares = [0, 0]
    for fill in fills.values():
        sign = 1 if orders[fill["order"]]["side"] == "buy" else -1
        shares[fill["day"] - 1] += sign * fill["quantity"]
        cash[fill["day"] - 1] -= sign * fill["quantity"] * fill["price"]
    owned = 70 + sum(shares)
    reservation = orders["S"]["remaining"]
    feasible = []
    for action in itertools.product(range(0, 51, 5), range(0, 81, 5), range(101)):
        state = settlement_repair(action, old_cash=cash, old_shares=shares, owned=owned, reservation=reservation)
        if state["feasible"]:
            feasible.append((-state["final_owned"], state["fee"], action, state))
    feasible.sort(key=lambda row: row[:3])
    negative_inventory, fee, action, state = feasible[0]
    certificate.update({"enumerated_actions": 11 * 17 * 101, "feasible_actions": len(feasible),
                        "surviving_fill_quantities": {key: fill["quantity"] for key, fill in fills.items()},
                        "accepted_remaining": {key: order["remaining"] for key, order in orders.items()},
                        "old_trade_shares_by_day": shares,
                        "maximum_final_inventory_action_count": sum(row[0] == negative_inventory for row in feasible),
                        "maximum_inventory_minimum_fee_actions": [row[2] for row in feasible if row[0] == negative_inventory and row[1] == fee],
                        "selected_constraint_slacks": {key: float(value) for key, value in state["slacks"].items()},
                        "tie_policy": "Maximum final owned shares, minimum exact fee, then action tuple."})
    return {"action": list(action), "bridge_fee": float(fee), "cash_path": list(map(float, state["cash"])),
            "final_owned_inventory": -negative_inventory, "inventory_after_T1": state["inventory1"],
            "live_sell_reservation": reservation, "old_trade_cash_by_day": cash,
            "owned_inventory_at_cutoff": owned, "settled_inventory_at_cutoff": 70}, certificate


CHECKS = {
    "c205": (c205, "Enumerate all deterministic two-query policies, including repeated fixed observations, with exact joint probabilities and optional stopping."),
    "c206": (c206, "Replay exchange-order fills and partial bust, preserve terminal cancellation and unacknowledged commitments, then exhaust funded whole-unit actions."),
    "c213": (c213, "Enumerate all 21,870 common whole-unit actions with exact implementation costs, two cash checkpoints and both borrowing bases; order every feasible action."),
    "c217": (c217, "Parse all 16 noisy joint cells and enumerate every budget-feasible adaptive two-query policy and terminal action with exact lexicographic ties."),
    "c218": (c218, "Parse corrected executions, reconcile ownership separately from dated settlements and commitments, then enumerate all 18,887 repair actions with exact financing fees."),
}
