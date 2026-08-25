ORDER_LOOKUP_TOOL = {
    "name": "order_lookup",
    "description": (
        "Look up the current status of a single customer order by order ID. "
        "Returns only customer-safe fields (never customer PII, risk scores, "
        "or internal notes). Call this only when the user is asking about a "
        "specific order and has supplied (or previously supplied in this "
        "conversation) an order ID. Do not guess or fabricate an order ID."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": (
                    "The order ID as given by the customer, "
                    "e.g. 'ORD-1007'. Pass it through as typed; "
                    "the tool normalizes formatting."
                ),
            }
        },
        "required": ["order_id"],
    },
}

ALL_TOOLS = [ORDER_LOOKUP_TOOL]