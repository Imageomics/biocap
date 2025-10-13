import pandas as pd
import argparse

def merge_results(
    unique_taxa_csv: str = "species_binomial_unique.csv",
    unique_scrape_csv: str = "descriptions_unique.csv",
    ambiguous_scrape_csv: str = "descriptions_ambiguous.csv",
    out_csv: str = "merged_descriptions.csv",
):

    tax_cols = ["binomial","kingdom","phylum","class","order","family","genus","species"]
    taxa_u = pd.read_csv(unique_taxa_csv)
    taxa_u = taxa_u[tax_cols].copy()
    for c in tax_cols:
        taxa_u[c] = taxa_u[c].astype(str).str.strip()
    taxa_u.replace({"nan": None, "": None}, inplace=True)
    taxa_u.dropna(subset=tax_cols, inplace=True)

    uniq = pd.read_csv(unique_scrape_csv)
    uniq = uniq.rename(columns={"species": "binomial"})
    uniq["binomial"] = uniq["binomial"].astype(str).str.strip()
    uniq["content"] = uniq["content"].astype(str)

    merged_u = taxa_u.merge(uniq, on="binomial", how="inner")

    amb = pd.read_csv(ambiguous_scrape_csv)
    keep_amb = tax_cols + ["content"]
    missing_amb = [c for c in keep_amb if c not in amb.columns]
    if missing_amb:
        raise ValueError(f"Ambiguous scrape CSV missing columns: {missing_amb}")
    merged_a = amb[keep_amb].copy()


    out_cols = tax_cols + ["content"]
    out_df = pd.concat([merged_u[out_cols], merged_a[out_cols]], ignore_index=True)

    for c in out_cols:
        out_df[c] = out_df[c].astype(str).str.strip()
    out_df.replace({"nan": None, "": None}, inplace=True)

    out_df.drop_duplicates(subset=["binomial"], keep="first", inplace=True)

    out_df.dropna(subset=tax_cols, inplace=True)

    out_df.to_csv(out_csv, index=False)
    print(f"Saved merged file with {len(out_df)} rows -> {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge unique and ambiguous Wikipedia scraping results")
    parser.add_argument("--unique-taxa", type=str, required=True, help="Input CSV with unique species taxonomy")
    parser.add_argument("--unique-descriptions", type=str, required=True, help="Scraped descriptions for unique species")
    parser.add_argument("--ambiguous-descriptions", type=str, required=True, help="Scraped descriptions for ambiguous species")
    parser.add_argument("--output", type=str, default="merged_descriptions.csv", help="Output merged CSV file")

    args = parser.parse_args()

    merge_results(
        unique_taxa_csv=args.unique_taxa,
        unique_scrape_csv=args.unique_descriptions,
        ambiguous_scrape_csv=args.ambiguous_descriptions,
        out_csv=args.output,
    )
