import polars as pl
from datetime import datetime
import os

RGBO_CSV_PATH = "rgbo.csv" # Path to the Rent Guidelines Board Orders CSV

def load_rgb_orders(csv_path=RGBO_CSV_PATH):
    """Loads RGBO data, parses dates using Polars, and prepares it for lookup."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"RGBO data file not found at: {csv_path}")

    try:
        # Use Polars read_csv
        df = pl.read_csv(csv_path)

        # Convert date columns to datetime objects using Polars expressions
        df = df.with_columns([
            pl.col('beginning_date').str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias('beginning_date'),
            pl.col('end_date').str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias('end_date')
        ])

        # Convert rate columns to numeric (Float64), coercing errors to null
        rate_cols = ['one_year_rate', 'two_year_rate', 'vacancy_lease_rate']
        for col in rate_cols:
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))
            else:
                print(f"Warning: Expected rate column '{col}' not found in {csv_path}. Filling with null.")
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col)) # Add column with nulls

        # Drop rows where essential dates are missing (null)
        df = df.drop_nulls(subset=['beginning_date', 'end_date'])
        # Sort by beginning_date descending
        df = df.sort(by='beginning_date', descending=True)

        print(f"Loaded {len(df)} RGBO records from {csv_path}")
        return df
    except Exception as e:
        print(f"Error loading or processing RGBO data from {csv_path}: {e}")
        raise

def get_rates_for_date(rgb_data, relevant_date):
    """
    Finds the applicable RGBO rates for a given date using Polars.

    Args:
        rgb_data (pl.DataFrame): DataFrame loaded by load_rgb_orders.
        relevant_date (datetime.date): The date to find rates for.

    Returns:
        dict or None: A dictionary containing rates for that date, or None if not found.
    """
    if relevant_date is None: # Check if date is None (previously pd.isna)
        return None

    # Convert relevant_date to the same type as DataFrame columns if needed, assuming date object
    relevant_date_pl = relevant_date # Polars can often compare directly with Python date/datetime

    # Find the first row where the relevant_date falls within the order's range
    applicable_order_df = rgb_data.filter(
        (pl.col('beginning_date') <= relevant_date_pl) &
        (pl.col('end_date') >= relevant_date_pl)
    )

    if applicable_order_df.height > 0:
        # Handle multiple matching orders if necessary (e.g., tenant vs owner paid heat)
        # For now, just return the first match as a dictionary.
        return applicable_order_df.row(0, named=True)
    else:
        return None

# --- Define Date Ranges from Flowchart ---
# Using date objects for comparison (Polars works well with date/datetime)
# Assuming the CSV dates are parsed as date objects by Polars' strptime
DATE_RANGES = {
    "range_83_97": (datetime(1983, 6, 19).date(), datetime(1997, 6, 18).date()),
    "range_97_11": (datetime(1997, 6, 19).date(), datetime(2011, 6, 23).date()),
    "range_11_15": (datetime(2011, 6, 24).date(), datetime(2015, 6, 14).date()),
    "range_15_19": (datetime(2015, 6, 15).date(), datetime(2019, 6, 13).date()),
    "range_19_present": (datetime(2019, 6, 14).date(), datetime.now().date()) # Use current date as end
}

def calculate_vacancy_allowance(unit_data, rgb_data):
    """
    Calculates the vacancy allowance based on the flowchart logic using Polars data.

    Args:
        unit_data (dict): A dictionary containing data for a single unit.
                          Expected keys: 'apartment_status', 'is_new_tenant',
                          'term_length', 'lease_start_date' (datetime.date obj),
                          'had_vacancy_allowance_in_prev_12_mo',
                          'previous_preferential_rent_has_value',
                          'tenant_tenure_years',
        rgb_data (pl.DataFrame): Loaded RGBO data (Polars DataFrame).

    Returns:
        float or str: The calculated vacancy allowance percentage (e.g., 0.05 for 5%)
                      or a descriptive string if a rate is indeterminable.
                      Returns None if input data is insufficient.
    """
    # --- Input Validation ---
    required_keys = ['apartment_status', 'is_new_tenant', 'term_length', 'lease_start_date',
                     'had_vacancy_allowance_in_prev_12_mo', 'previous_preferential_rent_has_value',
                     'tenant_tenure_years']
    if not all(key in unit_data for key in required_keys):
        print("Warning: Missing required keys in unit_data")
        return None
    # Use standard None check
    if unit_data['lease_start_date'] is None:
         print("Warning: Missing lease_start_date in unit_data")
         return None # Cannot proceed without lease start date

    # --- Get Relevant Rates ---
    # Convert lease_start_date to date object if it's datetime for comparison
    lease_start_dt = unit_data['lease_start_date']
    lease_start_date_obj = lease_start_dt.date() if isinstance(lease_start_dt, datetime) else lease_start_dt

    rates = get_rates_for_date(rgb_data, lease_start_date_obj) # Pass date object
    if rates is None:
        # Use f-string formatting or str() for the date
        return f"Indeterminable: No RGBO rates found for lease start date {str(lease_start_date_obj)}"

    # Rates is now a dictionary
    one_yr_rate = rates.get('one_year_rate', 0.0)
    two_yr_rate = rates.get('two_year_rate', 0.0)
    vac_lease_rate = rates.get('vacancy_lease_rate', 0.0) # May not exist in all orders

    # Fill None rates (Polars uses None for nulls) with 0.0
    one_yr_rate = 0.0 if one_yr_rate is None else one_yr_rate
    two_yr_rate = 0.0 if two_yr_rate is None else two_yr_rate
    vac_lease_rate = 0.0 if vac_lease_rate is None else vac_lease_rate


    # --- Flowchart Logic --- Helper function for duplicated logic ---
    def _calculate_term1_or_existing_allowance(lsd, unit_data, one_yr_rate, two_yr_rate, vac_lease_rate):
        """Helper function for logic shared by New Tenant (Term 1) and Existing Tenant."""
        if DATE_RANGES["range_83_97"][0] <= lsd <= DATE_RANGES["range_83_97"][1]:
             # Flowchart: [one_year_renewal_rate] - [vacancy_lease_rate]
             return one_yr_rate - vac_lease_rate
        elif DATE_RANGES["range_97_11"][0] <= lsd <= DATE_RANGES["range_97_11"][1]:
             # Flowchart: 20% - [two_year_renewal_rate] - [one_year_renewal]
             return 0.20 - two_yr_rate - one_yr_rate
        elif DATE_RANGES["range_11_15"][0] <= lsd <= DATE_RANGES["range_11_15"][1]:
            if unit_data['had_vacancy_allowance_in_prev_12_mo']:
                 if unit_data['previous_preferential_rent_has_value']:
                     # Flowchart: 20% - [two_year_renewal_rate] - [one_year_renewal]
                     return 0.20 - two_yr_rate - one_yr_rate
                 else:
                      # Flowchart: 20% - [two_year_renewal_rate]
                      return 0.20 - two_yr_rate
            else: # No vacancy allowance in prev 12 mo -> Tenant Tenure
                tenure = unit_data['tenant_tenure_years']
                if 0 <= tenure <= 2: return 0.05
                elif tenure == 3: return 0.10
                elif tenure == 4: return 0.15
                elif tenure > 4:
                    # Flowchart: 20% - [two_year_renewal_rate] - [one_year_renewal]
                    return 0.20 - two_yr_rate - one_yr_rate
                else: return "Indeterminable: Invalid Tenure"
        elif DATE_RANGES["range_15_19"][0] <= lsd <= DATE_RANGES["range_15_19"][1]:
             if unit_data['had_vacancy_allowance_in_prev_12_mo']:
                 if unit_data['previous_preferential_rent_has_value']:
                      # Flowchart: 20%
                       return 0.20
                 else:
                      # Flowchart: 20%
                      return 0.20
             else: # No vacancy allowance in prev 12 mo -> Tenant Tenure
                 tenure = unit_data['tenant_tenure_years']
                 if 0 <= tenure <= 2: return 0.05
                 elif tenure == 3: return 0.10
                 elif tenure == 4: return 0.15
                 elif tenure > 4: return 0.20
                 else: return "Indeterminable: Invalid Tenure"
        elif DATE_RANGES["range_19_present"][0] <= lsd <= DATE_RANGES["range_19_present"][1]:
            # Flowchart shows same outcome regardless of prev 12 mo allowance
            return one_yr_rate
        else:
             return "Indeterminable: Lease Start Date out of defined ranges"

    # --- Main Flowchart Logic ---
    lsd = lease_start_date_obj # Use the date object for comparisons

    if unit_data['apartment_status'] == "PE":
        return one_yr_rate

    # Assuming 'Not PE' leads to New Tenant check
    if unit_data['is_new_tenant']:
        # --- New Tenant Path ---
        if unit_data['term_length'] == 1:
            # --- Use Helper for Term Length 1 ---
            result = _calculate_term1_or_existing_allowance(lsd, unit_data, one_yr_rate, two_yr_rate, vac_lease_rate)
            # Append specific context to indeterminable result if needed
            return result if not isinstance(result, str) else f"{result} (New Tenant, Term 1)"

        elif unit_data['term_length'] >= 2:
             # --- Term Length 2+ ---
            if DATE_RANGES["range_83_97"][0] <= lsd <= DATE_RANGES["range_83_97"][1]:
                 # Flowchart: 20% - [two_year_renewal_rate] + [vacancy_lease_rate]
                 # Interpretation: Subtract two_yr, add vac_lease. CHECK THIS ASSUMPTION.
                 return 0.20 - two_yr_rate + vac_lease_rate
            elif DATE_RANGES["range_97_11"][0] <= lsd <= DATE_RANGES["range_97_11"][1]:
                 return 0.20
            elif DATE_RANGES["range_11_15"][0] <= lsd <= DATE_RANGES["range_11_15"][1]:
                 return 0.20
            elif DATE_RANGES["range_15_19"][0] <= lsd <= DATE_RANGES["range_15_19"][1]:
                if unit_data['had_vacancy_allowance_in_prev_12_mo']:
                     return one_yr_rate
                else: # No vacancy allowance in prev 12 mo -> Tenant Tenure
                     tenure = unit_data['tenant_tenure_years']
                     if 0 <= tenure <= 2: return 0.05
                     elif tenure == 3: return 0.10
                     elif tenure == 4: return 0.15
                     elif tenure > 4: return 0.20
                     else: return "Indeterminable: Invalid Tenure"
            elif DATE_RANGES["range_19_present"][0] <= lsd <= DATE_RANGES["range_19_present"][1]:
                 # Flowchart shows same outcome regardless of prev 12 mo allowance
                 return one_yr_rate
            else:
                 return "Indeterminable: Lease Start Date out of defined ranges (Term 2+)"
        else:
             return "Indeterminable: Invalid Term Length"
    else:
        # --- Existing Tenant Path ---
        # Use Helper for Existing Tenant (follows Term 1 logic per flowchart)
        result = _calculate_term1_or_existing_allowance(lsd, unit_data, one_yr_rate, two_yr_rate, vac_lease_rate)
        # Append specific context to indeterminable result if needed
        return result if not isinstance(result, str) else f"{result} (Existing Tenant)"


# --- Example Usage ---
if __name__ == "__main__":
    print("Loading RGBO data...")
    try:
        rgb_order_data = load_rgb_orders()

        # Example Unit Data (replace with actual data source)
        sample_units = [
            { # Example 1: PE Status
                'apartment_status': "PE", 'is_new_tenant': False, 'term_length': 1,
                'lease_start_date': datetime(2023, 10, 15).date(), # Use date object
                'had_vacancy_allowance_in_prev_12_mo': False,
                'previous_preferential_rent_has_value': False, 'tenant_tenure_years': 5
            },
            { # Example 2: New Tenant, Term 1, Date 2018, No Vac Allowance, Tenure 1yr
              'apartment_status': "RS", 'is_new_tenant': True, 'term_length': 1,
              'lease_start_date': datetime(2018, 7, 1).date(), # Use date object
              'had_vacancy_allowance_in_prev_12_mo': False,
              'previous_preferential_rent_has_value': False, 'tenant_tenure_years': 1
            },
             { # Example 3: New Tenant, Term 2+, Date 1990
              'apartment_status': "RS", 'is_new_tenant': True, 'term_length': 2,
              'lease_start_date': datetime(1990, 1, 1).date(), # Use date object
              'had_vacancy_allowance_in_prev_12_mo': False, # Doesn't matter for this branch
              'previous_preferential_rent_has_value': False, 'tenant_tenure_years': 0 # Doesn't matter
            },
             { # Example 4: Existing Tenant, Date 2012, Had Vac Allowance, Had Pref Rent
              'apartment_status': "RS", 'is_new_tenant': False, 'term_length': 1, # Term length irrelevant for existing?
              'lease_start_date': datetime(2012, 5, 5).date(), # Use date object
              'had_vacancy_allowance_in_prev_12_mo': True,
              'previous_preferential_rent_has_value': True, 'tenant_tenure_years': 6
            },
             { # Example 5: Missing Date
              'apartment_status': "RS", 'is_new_tenant': False, 'term_length': 1,
              'lease_start_date': None, # Use None instead of pd.NA
              'had_vacancy_allowance_in_prev_12_mo': True,
              'previous_preferential_rent_has_value': True, 'tenant_tenure_years': 6
            },


        ]

        print("\nCalculating vacancy allowances for sample units:")
        for i, unit in enumerate(sample_units):
            allowance = calculate_vacancy_allowance(unit, rgb_order_data)
            print(f"  Unit {i+1}: {allowance}")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred during example execution: {e}") 