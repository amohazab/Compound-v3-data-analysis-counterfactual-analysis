- **Compound_v3_analysis.ipynb:** The main jupyterhub notebook file that includes the analysis regarding the Compound v3protocol:

    - It Fetchs the necesary data in different steps (using get_data.py).
    
    - It executes treatment for different shortcomings of the data such as:
    
        - null values for some of the columns (market names)
    
        - adding loan value at the time of liquidation
        
        - adding collateral value usd and profit usd, since the current columns include incorrect values.
        
        - adding token prices usd
        
        - add parameters like store_front_pricefactor and liquidity factors to be used for estimating the collateral discount for liquidators
    
    - It performs data analysis for different values of interest such as LTV (Loan to Collateral ratio), collateral value and liquidation profit
    
    - It executes a counterfactual analysis by using a simple dynamic mechanism for close factor. At the end, it is concluded that the proposed mechanism improves the welfare of the users (borrowers) substantially while having a marginal effect on the liquidators, where only 5% of them make negative payoffs as the result of the new mechanism.

- **get_data.py:** It includes the functions we used the get the necessary data:

    - fetch_last_liquidations: first function getting the liquidation events from the compound v3 subgraph 
    (https://thegraph.com/explorer/subgraphs/AwoxEZbiWLvv6e3QdvdMZw4WDURdGbvPfHmZRc8Dpfz9?view=Query&chain=arbitrum-one)
    
    - add_positions_before: get borrowers loan and collateral values before liquidation using previous positions
    
    - fetch_prices_dataframe: get token proces at time of block numbers in the main dataframe.
    
    - get_comet_params_by_block:  get the necessary parameters to estimate the collateral discount such as liquidity factor and store_front_price_factor
 
      
- **data files:** Related data files in the analysis:
        - compound_v3_liqs.csv: The first driven file and the result of running the function fetch_last_liquidations. Note that no treatment has been added to this table and it includes data for all the pools
        - df_usdc_with_counterfactual.csv: It is the output file includeing all the added columns and treatments, but it only inlcudes observations related to the pool USDC-WETH.