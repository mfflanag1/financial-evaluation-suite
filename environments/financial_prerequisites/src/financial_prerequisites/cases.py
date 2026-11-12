"""Small synthetic workflow checks with trusted reference solutions.

Only ``prompt`` and ``public_files`` belong in a solver task. Expected outputs,
alternate inputs, and reference source remain in the trusted evaluator.
"""

from copy import deepcopy
import json
from textwrap import dedent


def _json(value):
    return json.dumps(value, indent=2, allow_nan=False) + "\n"


def _source(value):
    return dedent(value).lstrip()


_COMMON = """Work in /workspace using only Python's standard library and the supplied files.
Save your solution as /workspace/research.py, run it, and have it write
/workspace/policy.json. A prose answer is insufficient. The script takes no
arguments and runs with /workspace as its current directory. Do not install
packages or use the network.

Your saved script will also run against different public inputs with the same
schema in a fresh isolated directory. Read all values and identifiers from the
files: hardcoded fixture answers will fail. Emit exactly the stated JSON keys
and list ordering. JSON integers must be integers, never booleans or decimal
numbers; null must be JSON null. All numeric outputs must be finite. Other
numbers are compared with absolute tolerance 1e-9.\n"""


_SCHEMA_REFERENCE = _source('''
    import csv
    import json

    def solve():
        with open("contract.json") as handle:
            contract = json.load(handle)
        with open("observations.csv", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames
            rows = list(reader)
        amounts = [float(row["amount"]) for row in rows if row["amount"] != ""]
        start = contract["source_window_start"]
        end = contract["source_window_end"]
        return {
            "schema": {
                "columns": columns,
                "csv_row_count": len(rows),
                "entity_ids": sorted({row["entity_id"] for row in rows}),
                "amount": {"nonnull_count": len(amounts),
                           "null_count": len(rows) - len(amounts),
                           "sum": sum(amounts)},
                "eligible_true_count": sum(row["eligible"] == "true" for row in rows),
            },
            "source_window": {"start_index": start, "end_index": end,
                              "row_count": sum(start <= int(row["observation_index"]) <= end
                                               for row in rows)},
        }

    if __name__ == "__main__":
        with open("policy.json", "w") as handle:
            json.dump(solve(), handle, allow_nan=False)
''')


_RETURNS_REFERENCE = _source('''
    import csv
    import json

    def solve():
        with open("contract.json") as handle:
            contract = json.load(handle)
        with open("prices.csv", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assets = []
        weight_sum = 0.0
        for row in rows:
            weight = float(row["weight"])
            simple_return = float(row["end_price"]) / float(row["start_price"]) - 1
            assets.append({"asset_id": row["asset_id"], "return": simple_return,
                           "contribution": weight * simple_return})
            weight_sum += weight
        weighted_return = sum(asset["contribution"] for asset in assets)
        return {"assets": assets,
                "portfolio": {"weight_sum": weight_sum,
                              "weighted_return": weighted_return,
                              "ending_value": contract["initial_value"] * (1 + weighted_return)}}

    if __name__ == "__main__":
        with open("policy.json", "w") as handle:
            json.dump(solve(), handle, allow_nan=False)
''')


_REPAIR_REFERENCE = _source('''
    import csv
    import json
    from decimal import Decimal

    def solve():
        with open("contract.json") as handle:
            contract = json.load(handle)
        totals = {account_id: Decimal("0") for account_id in contract["account_ids"]}
        counts = {account_id: 0 for account_id in contract["account_ids"]}
        with open("transactions.csv", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["amount"] == "":
                    continue
                account_id = row["account_id"]
                totals[account_id] += Decimal(row["amount"].replace(",", ""))
                counts[account_id] += 1
        return {"accounts": [{"account_id": account_id,
                              "transaction_count": counts[account_id],
                              "net_amount": float(totals[account_id])}
                             for account_id in contract["account_ids"]],
                "total_net_amount": float(sum(totals.values()))}

    if __name__ == "__main__":
        with open("policy.json", "w") as handle:
            json.dump(solve(), handle, allow_nan=False)
''')


_BROKEN_STARTER = _source('''
    import csv
    import json

    with open("contract.json") as handle:
        contract = json.load(handle)
    totals = {account_id: 0.0 for account_id in contract["account_ids"]}
    counts = {account_id: 0 for account_id in contract["account_ids"]}
    with open("transactions.csv", newline="") as handle:
        for row in csv.reader(handle):
            account_id = row["account"]
            totals[account_id] += float(row["amount"])
            counts[account_id] += 1
    output = {"accounts": [{"account_id": account_id,
                            "transaction_count": counts[account_id],
                            "net_amount": totals[account_id]}
                           for account_id in contract["account_ids"]],
              "total_net_amount": sum(totals.values())}
    with open("policy.json", "w") as handle:
        json.dump(output, handle)
''')


_COVERAGE_REFERENCE = _source('''
    import csv
    import json

    def solve():
        with open("contract.json") as handle:
            contract = json.load(handle)
        with open("observations.csv", newline="") as handle:
            rows = list(csv.DictReader(handle))
        results = []
        for request in contract["requests"]:
            values = [float(row["value"]) for row in rows
                      if row["entity_id"] == request["entity_id"]
                      and request["start_index"] <= int(row["observation_index"]) <= request["end_index"]
                      and row["value"] != ""]
            if request["operation"] == "count":
                value = len(values)
            elif request["operation"] == "sum":
                value = sum(values)
            else:
                value = sum(values) / len(values) if values else None
            results.append({"request_id": request["request_id"],
                            "observation_count": len(values), "value": value})
        return {"results": results, "request_count": len(results)}

    if __name__ == "__main__":
        with open("policy.json", "w") as handle:
            json.dump(solve(), handle, allow_nan=False)
''')


def _request(request_id, entity_id, start, end, operation):
    return {"request_id": request_id, "entity_id": entity_id,
            "start_index": start, "end_index": end, "operation": operation}


_CASES = [
    {
        "id": "prerequisite-schema-v1",
        "kind": "schema_inspection",
        "prompt": _COMMON + """
Inspect observations.csv and contract.json. CSV columns are entity_id (string,
including leading zeros), observation_index (integer), amount (number or an
empty cell meaning null), and eligible (the literal true or false). Preserve
column order. The contract has FLAT fields source_window_start and
source_window_end; they define an inclusive observation-index window. Count
every CSV data row, including rows with null amounts. Amount statistics and
eligible counts use the entire CSV; only source_window.row_count uses the window.

Write exactly:
{"schema":{"columns":[column names in file order],"csv_row_count":integer,
"entity_ids":[unique string IDs in lexicographic order],
"amount":{"nonnull_count":integer,"null_count":integer,"sum":number},
"eligible_true_count":integer},
"source_window":{"start_index":integer,"end_index":integer,"row_count":integer}}.
The amount sum ignores nulls. No CSV header is a data row.
""",
        "public_files": {
            "contract.json": _json({"source_window_start": 1, "source_window_end": 2}),
            "observations.csv": "entity_id,observation_index,amount,eligible\n001,0,10,true\n010,1,,false\n001,2,-2.5,true\nB,3,0,false\n010,2,7.5,true\nB,1,5,false\n",
        },
        "expected_json": {
            "schema": {"columns": ["entity_id", "observation_index", "amount", "eligible"],
                       "csv_row_count": 6, "entity_ids": ["001", "010", "B"],
                       "amount": {"nonnull_count": 5, "null_count": 1, "sum": 20.0},
                       "eligible_true_count": 3},
            "source_window": {"start_index": 1, "end_index": 2, "row_count": 4},
        },
        "alternate_public_files": {
            "contract.json": _json({"source_window_start": 0, "source_window_end": 0}),
            "observations.csv": "entity_id,observation_index,amount,eligible\n009,0,,false\nZ,1,4,true\n009,0,-1,true\n02,2,0,false\n",
        },
        "alternate_expected_json": {
            "schema": {"columns": ["entity_id", "observation_index", "amount", "eligible"],
                       "csv_row_count": 4, "entity_ids": ["009", "02", "Z"],
                       "amount": {"nonnull_count": 3, "null_count": 1, "sum": 3.0},
                       "eligible_true_count": 2},
            "source_window": {"start_index": 0, "end_index": 0, "row_count": 2},
        },
        "reference_source": _SCHEMA_REFERENCE,
    },
    {
        "id": "prerequisite-calculation-v1",
        "kind": "saved_calculation",
        "prompt": _COMMON + """
Read prices.csv (asset_id, start_price, end_price, weight) and contract.json
(initial_value). For each asset calculate its simple return as
end_price/start_price - 1, and contribution as weight * simple return. Weights
are supplied as fractions and must not be normalized. Compute weight_sum,
weighted_return (sum of contributions), and ending_value as
initial_value * (1 + weighted_return). Prices are positive and there are no
costs, distributions, or missing cells. Preserve the CSV asset order.

Write exactly:
{"assets":[{"asset_id":string,"return":number,"contribution":number},...],
"portfolio":{"weight_sum":number,"weighted_return":number,"ending_value":number}}.
""",
        "public_files": {
            "contract.json": _json({"initial_value": 10000.0}),
            "prices.csv": "asset_id,start_price,end_price,weight\nA,100,110,0.5\nB,80,40,0.25\nC,25,50,0.25\n",
        },
        "expected_json": {
            "assets": [{"asset_id": "A", "return": 0.1, "contribution": 0.05},
                       {"asset_id": "B", "return": -0.5, "contribution": -0.125},
                       {"asset_id": "C", "return": 1.0, "contribution": 0.25}],
            "portfolio": {"weight_sum": 1.0, "weighted_return": 0.175, "ending_value": 11750.0},
        },
        "alternate_public_files": {
            "contract.json": _json({"initial_value": 2500.0}),
            "prices.csv": "asset_id,start_price,end_price,weight\nY,48,60,0.2\nX,25,20,0.3\nW,10,11,0.5\n",
        },
        "alternate_expected_json": {
            "assets": [{"asset_id": "Y", "return": 0.25, "contribution": 0.05},
                       {"asset_id": "X", "return": -0.2, "contribution": -0.06},
                       {"asset_id": "W", "return": 0.1, "contribution": 0.05}],
            "portfolio": {"weight_sum": 1.0, "weighted_return": 0.04, "ending_value": 2600.0},
        },
        "reference_source": _RETURNS_REFERENCE,
    },
    {
        "id": "prerequisite-repair-v1",
        "kind": "coding_error_recovery",
        "prompt": _COMMON + """
starter.py is a broken attempt to aggregate transactions.csv. Run it to inspect
the failure, then save a corrected solution as research.py. The CSV has
account_id and amount columns. Quoted amounts may contain thousands commas;
empty amounts are missing and excluded from both sums and transaction counts.
Each nonempty amount is one transaction, even when its amount is zero. The
contract's account_ids array specifies every output account and its exact
order; include accounts with no transactions. All CSV accounts are declared.

Write exactly:
{"accounts":[{"account_id":string,"transaction_count":integer,"net_amount":number},...],
"total_net_amount":number}.
The total is the sum of all account net amounts. Only the corrected script and
its outputs are scored; the initial error is diagnostic evidence.
""",
        "public_files": {
            "contract.json": _json({"account_ids": ["acct_c", "acct_a", "acct_b", "acct_empty"]}),
            "transactions.csv": 'account_id,amount\nacct_a,"1,200.50"\nacct_b,-25.25\nacct_a,5\nacct_c,\nacct_b,0\nacct_c,10\n',
            "starter.py": _BROKEN_STARTER,
        },
        "expected_json": {
            "accounts": [{"account_id": "acct_c", "transaction_count": 1, "net_amount": 10.0},
                         {"account_id": "acct_a", "transaction_count": 2, "net_amount": 1205.5},
                         {"account_id": "acct_b", "transaction_count": 2, "net_amount": -25.25},
                         {"account_id": "acct_empty", "transaction_count": 0, "net_amount": 0.0}],
            "total_net_amount": 1190.25,
        },
        "alternate_public_files": {
            "contract.json": _json({"account_ids": ["group_z", "group_y", "group_x"]}),
            "transactions.csv": 'account_id,amount\ngroup_x,"-2,000.25"\ngroup_y,1.25\ngroup_x,"2,100.00"\ngroup_y,\ngroup_y,0\n',
            "starter.py": _BROKEN_STARTER,
        },
        "alternate_expected_json": {
            "accounts": [{"account_id": "group_z", "transaction_count": 0, "net_amount": 0.0},
                         {"account_id": "group_y", "transaction_count": 2, "net_amount": 1.25},
                         {"account_id": "group_x", "transaction_count": 2, "net_amount": 99.75}],
            "total_net_amount": 101.0,
        },
        "reference_source": _REPAIR_REFERENCE,
    },
    {
        "id": "prerequisite-coverage-v1",
        "kind": "structured_coverage",
        "prompt": _COMMON + """
Read observations.csv (entity_id, observation_index, value) and every request
in contract.json. For each request, select rows for its entity_id whose integer
observation_index lies within start_index and end_index, inclusive. Empty value
cells are missing; exclude them from the observation count and calculations.
operation is count, sum, or mean. count returns a JSON integer; sum of no values
is zero; mean of no values is JSON null. Entities with no rows are valid requests.
Include every request exactly once, in the contract's request order.

Write exactly:
{"results":[{"request_id":string,"observation_count":integer,"value":number-or-null},...],
"request_count":integer}.
For count requests value must specifically be an integer. request_count is the
number of requested results, including empty selections. Use no booleans.
""",
        "public_files": {
            "contract.json": _json({"requests": [
                _request("r01", "A", 0, 2, "count"),
                _request("r02", "A", 0, 2, "mean"),
                _request("r03", "B", 1, 3, "sum"),
                _request("r04", "C", 0, 4, "mean"),
                _request("r05", "C", 0, 4, "count"),
                _request("r06", "A", 1, 1, "sum"),
            ]}),
            "observations.csv": "entity_id,observation_index,value\nA,0,0\nA,1,\nA,2,6\nB,0,100\nB,1,-2\nB,2,3.5\nB,3,4.5\n",
        },
        "expected_json": {
            "results": [{"request_id": "r01", "observation_count": 2, "value": 2},
                        {"request_id": "r02", "observation_count": 2, "value": 3.0},
                        {"request_id": "r03", "observation_count": 3, "value": 6.0},
                        {"request_id": "r04", "observation_count": 0, "value": None},
                        {"request_id": "r05", "observation_count": 0, "value": 0},
                        {"request_id": "r06", "observation_count": 0, "value": 0.0}],
            "request_count": 6,
        },
        "alternate_public_files": {
            "contract.json": _json({"requests": [
                _request("s07", "Y", 2, 4, "mean"),
                _request("s03", "Z", 0, 1, "sum"),
                _request("s02", "Y", 1, 1, "count"),
                _request("s08", "X", 0, 4, "mean"),
                _request("s01", "Y", 0, 4, "count"),
                _request("s09", "Z", 2, 2, "mean"),
                _request("s04", "X", 0, 4, "sum"),
            ]}),
            "observations.csv": "entity_id,observation_index,value\nY,1,0\nY,2,\nY,3,-4\nY,4,8\nZ,0,1.25\nZ,1,2.75\nZ,2,\n",
        },
        "alternate_expected_json": {
            "results": [{"request_id": "s07", "observation_count": 2, "value": 2.0},
                        {"request_id": "s03", "observation_count": 2, "value": 4.0},
                        {"request_id": "s02", "observation_count": 1, "value": 1},
                        {"request_id": "s08", "observation_count": 0, "value": None},
                        {"request_id": "s01", "observation_count": 3, "value": 3},
                        {"request_id": "s09", "observation_count": 0, "value": None},
                        {"request_id": "s04", "observation_count": 0, "value": 0.0}],
            "request_count": 7,
        },
        "reference_source": _COVERAGE_REFERENCE,
    },
]


def load_cases():
    """Return independent copies; keep trusted fields out of solver task data."""
    return deepcopy(_CASES)
