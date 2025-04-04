dolmino = {
    "DCLM": 0.472,
    "Flan": 0.166,
    "PES2O": 0.0585,
    "Wiki": 0.0711,
    "Stackexchange": 0.0245,
    "Math": 0.208,
}

# available_data = 50_000_000_000
# number_of_tokens = 5_000_000_000
available_data = 10_001_000
number_of_tokens = 5_000_000

for k, v in dolmino.items():
    # compute proportion of data needed to subsample
    token_budget = v * available_data
    token_needed = v * number_of_tokens
    ratio = token_needed / token_budget
    print(f"{k}: {ratio}")

available_data_pdf = 10_000_000
number_of_tokens_pdf = 5_000_000
token_budget_pdf = available_data_pdf / number_of_tokens_pdf
token_needed_pdf = number_of_tokens_pdf / available_data_pdf
ratio_pdf = token_needed_pdf / token_budget_pdf
print(f"PDF: {ratio_pdf}")