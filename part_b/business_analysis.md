# Business Case Analysis — Promotion Effectiveness at a Fashion Retail Chain

---

## B1. Problem Formulation

### B1(a) — Machine Learning Problem Formulation

**Target Variable:** `items_sold` — the number of items sold at a store in a given month under a specific promotion.

**Candidate Input Features:**
- Store features: `store_size`, `location_type` (urban/semi-urban/rural), `competition_density`, monthly footfall
- Promotion features: `promotion_type` (Flat Discount, BOGO, Free Gift, Category-Specific Offer, Loyalty Points Bonus)
- Time features: `month`, `is_weekend`, `is_festival`
- Customer demographic features: average age group, income level of the store's catchment area

**Type of ML Problem:** This is a **supervised regression problem**.

**Justification:** The target variable `items_sold` is a continuous numerical value. We have historical labelled data — past records of which promotion was run at which store and how many items were sold. Since we want to *predict a quantity* (items sold), regression is the correct problem type. If instead we only wanted to classify which promotion is "best" (a ranking), it could also be framed as a multi-class classification problem, but regression gives richer, quantitative output that is more useful for business decision-making.

---

### B1(b) — Why Items Sold is a Better Target than Revenue

**Revenue = Price × Quantity.** Revenue is affected by both the number of items sold *and* the price at which they are sold. Promotions like Flat Discounts lower the price directly, which can make revenue look lower even when more items are being moved. This creates a misleading signal — a promotion that dramatically increases sales volume might appear to "underperform" just because the unit price dropped.

**Items sold (sales volume)** removes the pricing effect and gives a clean measure of whether the promotion is actually driving customer purchase behaviour. It is not distorted by discounts, price changes, or product mix shifts.

**Broader Principle:** Target variable selection should reflect the *true business goal* and be robust to confounding factors. A good target variable is one that (1) directly measures what the business wants to optimise, and (2) is not inflated or deflated by variables outside the model's control. Choosing a noisy or proxy target variable leads to models that optimise the wrong thing — this is called "metric misalignment."

---

### B1(c) — Alternative to a Single Global Model

**Problem with a single global model:** A single model trained on all 50 stores assumes all stores behave similarly. In reality, a Flat Discount may work very well in a rural store where customers are price-sensitive, while Loyalty Points Bonus may work better in an urban store with a repeat customer base. A global model will average out these differences and give suboptimal recommendations for individual stores.

**Proposed Alternative — Stratified or Hierarchical Modelling Strategy:**

1. **Segmented Models by Location Type:** Train separate models for urban, semi-urban, and rural stores. Each model learns promotion response patterns specific to that location type.

2. **Store-Cluster Based Models:** Use clustering (e.g., K-Means) to group stores by similar characteristics (size, footfall, competition density), then train one model per cluster. This is more data-efficient than one model per store.

3. **Mixed Effects / Hierarchical Model:** Use a single model that includes store-level fixed effects or store ID as a feature, allowing the model to learn both global patterns and store-specific deviations.

**Justification:** These strategies respect the heterogeneity across stores while still pooling enough data to train reliable models. Stratified models are easy to interpret and maintain, making them practical for a retail business team.

---

## B2. Data and EDA Strategy

### B2(a) — Joining the Four Tables

**The four tables are:**
1. `transactions` — one row per transaction (transaction_id, store_id, date, items_sold, promotion_id)
2. `store_attributes` — one row per store (store_id, store_size, location_type, competition_density, footfall)
3. `promotion_details` — one row per promotion (promotion_id, promotion_type)
4. `calendar` — one row per date (date, is_weekend, is_festival, month, year)

**Join Strategy:**
- Join `transactions` with `store_attributes` on `store_id` → adds store context to each transaction
- Join the result with `promotion_details` on `promotion_id` → adds promotion type
- Join with `calendar` on `date` → adds time-based flags like `is_weekend`, `is_festival`, `month`

All joins should be **left joins** from the transactions table, so no transaction records are lost.

**Grain of the Final Dataset:** One row = **one store × one month × one promotion type**

**Aggregations Before Modelling:**
- Sum `items_sold` per store per month (the target variable)
- Average or mode of `promotion_type` per store per month (if promotions vary within a month)
- Count of festival days and weekend days per month per store
- Average `competition_density` and `footfall` per store per month

---

### B2(b) — EDA Strategy Before Modelling

**Analysis 1 — Distribution of Items Sold (Target Variable)**
- Plot a histogram of `items_sold`
- Look for: skewness, outliers, whether a log transformation is needed
- Impact: If highly skewed, apply log transformation to stabilise variance before modelling

**Analysis 2 — Items Sold by Promotion Type (Box Plot)**
- Plot a box plot of `items_sold` for each of the 5 promotion types
- Look for: which promotion type has the highest median sales, and how much overlap exists between promotions
- Impact: If one promotion clearly dominates, the model should have `promotion_type` as a strong feature; if there is little difference, other features matter more

**Analysis 3 — Items Sold by Location Type (Bar or Box Plot)**
- Compare average `items_sold` across urban, semi-urban, and rural stores
- Look for: significant differences in baseline sales by location
- Impact: Confirms whether a single global model is appropriate or whether location-segmented models are needed

**Analysis 4 — Correlation Heatmap of Numerical Features**
- Plot a correlation heatmap between `items_sold`, `footfall`, `competition_density`, `store_size` (encoded), and time features
- Look for: highly correlated input features (multicollinearity) and features with strong correlation to the target
- Impact: Drop or combine highly correlated input features; prioritise high-correlation features for modelling

**Analysis 5 — Seasonal Trend of Items Sold (Line Plot Over Time)**
- Plot average `items_sold` by month across all stores
- Look for: seasonal peaks (e.g., December festivals, sale season), trend over time
- Impact: Confirms that `month` and `is_festival` should be included as features; reveals if the data has non-stationarity that affects train-test splitting

---

### B2(c) — Handling the 80% No-Promotion Imbalance

**How it affects the model:**
If 80% of records have no promotion, the model will see very few examples of promotion-driven sales patterns during training. It will be biased towards predicting "no promotion" behaviour and may underestimate the impact of promotions. The model may not learn meaningful differences *between* promotion types because there are too few examples relative to the no-promotion baseline.

**Steps to Address This:**

1. **Separate Analysis First:** Analyse no-promotion and promotion periods separately in EDA to understand their baseline differences before combining them.

2. **Oversample Promotion Records:** Use techniques like SMOTE (if classification) or simple row duplication/bootstrapping of promotion records to give the model more promotion examples during training.

3. **Include a Binary Flag Feature:** Add a feature `has_promotion` (1 = yes, 0 = no). This helps the model learn to distinguish promotion vs. non-promotion behaviour explicitly.

4. **Stratified Sampling:** When splitting train/test, ensure both sets contain a representative proportion of promotion types so the test set reflects all scenarios.

5. **Evaluate on Promotion-Only Subset:** In addition to overall metrics, separately evaluate model performance only on promotion records to ensure it is learning promotion effects accurately.

---

## B3. Model Evaluation and Deployment

### B3(a) — Train-Test Split Strategy and Evaluation Metrics

**Why Random Split is Inappropriate:**
This data is time-ordered (monthly records across 3 years). A random split would allow future data to "leak" into the training set — e.g., the model might train on December 2024 records and test on March 2024. This creates **data leakage**, where the model learns patterns from the future that it would not have access to in real deployment. The result is an overly optimistic evaluation that fails in production.

**Correct Approach — Temporal Split:**
- Sort data chronologically by month and year
- Use the first 2 years (24 months) as the **training set**
- Use the final 12 months (most recent year) as the **test set**
- This mirrors real-world deployment: train on the past, predict the future

**Evaluation Metrics:**

| Metric | Formula | Business Interpretation |
|--------|---------|------------------------|
| **RMSE** (Root Mean Squared Error) | √(mean of squared errors) | Penalises large prediction errors heavily. Important here because badly over/under-ordering stock based on wrong predictions is costly. |
| **MAE** (Mean Absolute Error) | Mean of absolute errors | Average number of items the prediction is off by. Easy to explain to the marketing team — "on average, we are off by X items per store per month." |
| **R² Score** | 1 - SS_res/SS_tot | Measures how much variance in items_sold the model explains. An R² of 0.80 means the model explains 80% of the variation — a useful headline metric for stakeholders. |

---

### B3(b) — Investigating Different Recommendations for the Same Store

**Scenario:** Store 12 gets Loyalty Points Bonus in December but Flat Discount in March.

**How to Investigate Using Feature Importance:**

1. **Extract Feature Importances from the Model:** Use the trained Random Forest or Gradient Boosting model to extract feature importance scores. Features like `month`, `is_festival`, `promotion_type`, and `footfall` will each have an assigned importance.

2. **Compare Input Feature Values for Store 12 in December vs. March:**

| Feature | December | March |
|---------|----------|-------|
| `is_festival` | 1 (Christmas/New Year) | 0 |
| `month` | 12 | 3 |
| `footfall` | High | Moderate |
| `competition_density` | Low | Medium |

3. **Use SHAP Values (if available):** SHAP (SHapley Additive exPlanations) can show *exactly how much* each feature pushed the prediction towards one promotion over another for a specific store-month. This gives a clear, feature-level explanation.

**Communicating to the Marketing Team:**

Present a simple table or bar chart showing:
- *"In December, the model chose Loyalty Points Bonus because footfall is 40% higher, there is a festival flag, and customers are more likely to return — making loyalty rewards effective."*
- *"In March, footfall is lower and there is no festival. The model finds that Flat Discount creates the biggest uplift in this low-footfall period by attracting price-sensitive customers."*

Avoid technical jargon. Frame it in business language: seasons, customer behaviour, and competition — not model coefficients.

---

### B3(c) — End-to-End Deployment for Monthly Recommendations

**Step 1 — Save the Trained Model**
After training, save the entire pipeline (preprocessing + model) using Python's `joblib` library:
```python
import joblib
joblib.dump(pipeline, 'promotion_recommender_v1.pkl')
```
This saves both the scaler and the model in one file so new data is processed consistently.

**Step 2 — Prepare New Monthly Data**
At the start of each month:
- Collect store attributes (footfall, competition density) for the upcoming month
- Add calendar features: `month`, `year`, `is_festival`, `is_weekend` count for the month
- Format the data in exactly the same column structure as the training data
- Do **not** retrain the scaler or encoder — use the saved pipeline which already has fitted transformers

**Step 3 — Generate Recommendations**
Load the saved pipeline and run predictions:
```python
model = joblib.load('promotion_recommender_v1.pkl')
predictions = model.predict(new_month_data)
```
For each of the 50 stores, the model outputs predicted `items_sold` for each of the 5 promotion types. Recommend the promotion with the highest predicted items_sold for each store.

**Step 4 — Monitoring for Model Degradation**
Set up the following checks each month after actual results come in:

| Monitoring Check | What to Watch | Action if Alert Triggers |
|-----------------|--------------|--------------------------|
| **Prediction vs. Actual Tracking** | Compare predicted items_sold vs. actual items_sold each month; track MAE over time | If MAE increases by >20% over 3 consecutive months, trigger retraining |
| **Feature Distribution Drift** | Monitor if input features (footfall, competition_density) shift significantly from training distribution using statistical tests (e.g., KS test) | If drift detected, flag for data review and possible retraining |
| **Business Metric Monitoring** | Track actual sales uplift from recommended promotions month over month | If uplift starts declining, model may no longer reflect current customer behaviour |
| **Scheduled Retraining** | Retrain model every 6 months regardless of performance | Ensures model stays updated with recent seasonal patterns |

**Retraining Trigger:** When any monitoring check fires, retrain the model on the most recent 24 months of data (rolling window), re-evaluate on the latest 6-month hold-out, and redeploy if performance improves.