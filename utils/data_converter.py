def convert_data(df):
    df["currency_id"] = df["currency_id"] == "ARS"
    df["condition"] = df["condition"] == "new"

    df["warranty"] = df["warranty"].notna()
    df["video_id"] = df["video_id"].notna()
    df["official_store_id"] = df["official_store_id"].notna()

    df["deal_ids"] = df["deal_ids"].apply(lambda x: x != [])
    df["variations"] = df["variations"].apply(lambda x: x != [])
    df["attributes"] = df["attributes"].apply(lambda x: x != [])

    df["tags"] = df["tags"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "No Tags")
    df["sub_status"] = df["sub_status"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "No Sub Status")

    df["local_pickup"] = df["shipping"].apply(lambda x: x.get("local_pick_up") if isinstance(x, dict) else None)
    df["free_shipping"] = df["shipping"].apply(lambda x: x.get("free_shipping") if isinstance(x, dict) else None)
    df["has_dimensions"] = df["shipping"].apply(lambda x: x.get("dimensions") is not None if isinstance(x, dict) else False)

    def normalizar_desc(desc):
        return desc.lower().replace(" ", "_")

    kind_payment = set()
    for methods in df["non_mercado_pago_payment_methods"]:
        if isinstance(methods, list):
            for method in methods:
                desc = method.get("description")
                if desc:
                    kind_payment.add(normalizar_desc(desc))

    for desc_norm in kind_payment:
        df[desc_norm] = df["non_mercado_pago_payment_methods"].apply(
            lambda lst: any(normalizar_desc(d.get("description", "")) == desc_norm for d in lst) if isinstance(lst, list) else False
        )

    for col in kind_payment:
        if df[col].nunique() == 1:
            df.drop(columns=[col], inplace=True)

    return df, kind_payment
