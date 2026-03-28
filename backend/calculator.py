#This code implements our FIRE model using real-world financial research, with a focus on Indian factors like inflation and market returns. It calculates the retirement fund needed and monthly investments by applying principles like compound growth and risk-based returns. It also offers practical insights, including savings analysis, asset allocation, and financial safety measures to help users achieve early retirement
def calculate_fire(data):
    """
    data = FIREInput object
    Returns enhanced FIRE insights
    """

    
    years_to_fire  = data.fire_target_age - data.age
    months_to_fire = years_to_fire * 12

    
    inflation_rate = 0.06  

    future_monthly_expenses = data.monthly_expenses_post_fire * ((1 + inflation_rate) ** years_to_fire)
    annual_post_fire_expenses = future_monthly_expenses * 12

    
    fire_corpus_min = annual_post_fire_expenses * 25
    fire_corpus_max = annual_post_fire_expenses * 35

   
    fire_corpus_min *= 1.1
    fire_corpus_max *= 1.1

    
    return_rates = {
        "conservative": 0.08,
        "moderate":     0.11,
        "aggressive":   0.14
    }

    annual_return  = return_rates.get(str(data.risk_profile.value), 0.11)
    monthly_return = annual_return / 12

   
    total_existing = data.current_savings + data.existing_investments

    existing_future_value = total_existing * ((1 + annual_return) ** years_to_fire)


    remaining_min = max(fire_corpus_min - existing_future_value, 0)
    remaining_max = max(fire_corpus_max - existing_future_value, 0)

   
    def calculate_sip(fv):
        if monthly_return > 0 and months_to_fire > 0:
            return fv * monthly_return / ((1 + monthly_return) ** months_to_fire - 1)
        return fv / months_to_fire if months_to_fire > 0 else 0

    monthly_sip_min = calculate_sip(remaining_min)
    monthly_sip_max = calculate_sip(remaining_max)

   
    monthly_savings = data.monthly_income - data.monthly_expenses
    savings_rate    = (monthly_savings / data.monthly_income) * 100 if data.monthly_income > 0 else 0

    if savings_rate < 20:
        savings_insight = "Low savings rate — FIRE will take longer"
    elif savings_rate < 40:
        savings_insight = "Good savings rate — you are on track"
    else:
        savings_insight = "Excellent savings rate — fast FIRE possible"

   
    emergency_fund_needed = data.monthly_expenses * 6

   
    if data.monthly_expenses_post_fire < 30000:
        fire_type = "Lean FIRE"
    elif data.monthly_expenses_post_fire < 100000:
        fire_type = "Moderate FIRE"
    else:
        fire_type = "Fat FIRE"

  
    allocations = {
        "conservative": {
            "Large Cap Funds": 30,
            "Debt Funds":      40,
            "PPF":             20,
            "Gold":            10
        },
        "moderate": {
            "Large Cap": 40,
            "Mid Cap":   20,
            "Debt":      25,
            "Gold":      10,
            "REITs":      5
        },
        "aggressive": {
            "Large Cap": 35,
            "Mid Cap":   25,
            "Small Cap": 25,
            "Gold":       5,
            "REITs":     10
        }
    }

  
    return {
        "fire_corpus_range": {
            "min": round(fire_corpus_min, 2),
            "max": round(fire_corpus_max, 2)
        },
        "monthly_sip_range": {
            "min": round(monthly_sip_min, 2),
            "max": round(monthly_sip_max, 2)
        },
        "years_to_fire": years_to_fire,
        "monthly_savings": round(monthly_savings, 2),
        "savings_rate_percent": round(savings_rate, 1),
        "savings_insight": savings_insight,
        "projected_existing_corpus": round(existing_future_value, 2),
        "emergency_fund_needed": round(emergency_fund_needed, 2),
        "fire_type": fire_type,
        "asset_allocation": allocations.get(
            str(data.risk_profile.value),
            allocations["moderate"]
        ),
        "tax_suggestions": [
            "Use Section 80C (PPF, ELSS)",
            "Invest in NPS for additional deduction",
            "Hold investments long-term for tax efficiency"
        ],
        "insurance_advice": {
            "term_insurance": round(data.monthly_income * 12 * 15, 2),
            "health_insurance": "₹10–20 lakh coverage recommended"
        },
        "warnings": [
            "Market returns are not guaranteed",
            "Inflation can reduce purchasing power",
            "Healthcare costs may rise significantly"
        ]
    }