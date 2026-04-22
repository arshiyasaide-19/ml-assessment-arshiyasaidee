# Part B: Business Case Analysis
## Promotion Effectiveness at a Fashion Retail Chain

---

## B1. Problem Formulation

### B1(a) — What Kind of ML Problem Is This?

**Target Variable (what we want to predict):**
`items_sold` — the number of items sold in a store when a promotion is running.

**Input Features (information we give the model to learn from):**
- Store details: store size (small/medium/large), location type 
  (urban/semi-urban/rural), competition nearby
- Promotion type: Flat Discount, BOGO, Free Gift, 
  Category-Specific, Loyalty Points
- Time info: which month, is it a weekend, is it a festival day
- Customer info: footfall (how many people visit the store)

**Type of ML Problem:**
This is a **Regression Problem** — because we are predicting a 
number (items sold), not a category.

After predicting, we compare all 5 promotion types for each store 
and pick the one with the highest predicted items_sold. 
That becomes our recommendation.

---

### B1(b) — Why Use Items Sold Instead of Revenue?

Revenue = items sold × price.

The problem with using revenue is:
- A Flat Discount lowers the price, so revenue looks lower 
  even if MORE items were sold
- This makes discount promotions look bad even when they 
  are actually working well

**Items sold is better because:**
- It directly tells us if a promotion made more people buy
- It is not affected by price changes
- It measures the true effect of the promotion

**The bigger lesson:**
Always choose a target variable that directly measures 
what you want to improve — not a side effect of it.

---

### B1(c) — One Model for All Stores? Bad Idea!

A junior analyst suggests using ONE model for all 50 stores.

**Why this is a problem:**
A store in a busy Mumbai mall behaves very differently from 
a small rural store. If we train one model on all stores together, 
it will average out all the differences and give mediocre 
recommendations to everyone.

**Better approach — Group Similar Stores Together:**
1. Use clustering (like K-Means) to group stores that are similar
   (e.g., all large urban stores in one group, 
   small rural stores in another)
2. Train a separate model for each group
3. Each model learns what works best for stores like those

This way, recommendations are tailored and more accurate.

---

## B2. Data and EDA Strategy

### B2(a) — How to Join the 4 Tables

We have 4 separate tables. We need to combine them into 
one big table before building a model.

**The 4 Tables:**
| Table | What It Contains |
|-------|-----------------|
| Transactions | Date, store ID, promotion used, items sold |
| Store Attributes | Store ID, size, location, footfall, competition |
| Promotion Details | Promotion name, discount amount, category |
| Calendar | Date, is_weekend, is_festival |

**How to Join Them:**
1. Join Transactions + Store Attributes → using `store_id`
2. Join result + Promotion Details → using `promotion_type`
3. Join result + Calendar → using `date`

**Final Table Grain (what each row means):**
One row = one store + one month + one promotion type

So we group (aggregate) daily data into monthly totals:
- Total `items_sold` per store per month per promotion
- Count of festival days in that month
- Count of weekend days in that month

---

### B2(b) — EDA: 4 Charts We Would Make

Before building a model, we always explore the data visually.

**1. Box Plot — Items Sold by Promotion Type**
- What we look for: Which promotion sells the most items?
- Why it matters: If BOGO always wins, we might not need 
  a complex model

**2. Grouped Bar Chart — Location Type vs Promotion Type**
- What we look for: Does Loyalty Points work better in 
  rural areas? Does Flat Discount work better in cities?
- Why it matters: Confirms we need location-specific models

**3. Line Chart — Monthly Sales Trend**
- What we look for: Are sales higher in December? 
  Do festivals boost sales?
- Why it matters: We need to add month and festival 
  features to our model

**4. Correlation Heatmap**
- What we look for: Which features (footfall, competition, 
  store size) are most related to items_sold?
- Why it matters: Helps us pick the most useful features 
  for the model

---

### B2(c) — 80% Transactions Have No Promotion — Problem!

**The problem:**
If most of our data has no promotion, the model will mostly 
learn "normal sales without promotions" and will not 
understand promotions well.

**How to fix it:**

1. **Focus only on promoted transactions** when training 
   the promotion recommendation model

2. **Calculate a lift score instead:**
   Lift = items sold WITH promotion ÷ average items sold 
   WITHOUT promotion
   This shows how much EXTRA sales each promotion creates

3. **Give more importance to promoted rows** during training 
   by using sample weights, so the model pays more 
   attention to promotion patterns

---

## B3. Model Evaluation and Deployment

### B3(a) — How to Split Train and Test Data

**The right way — Time-Based Split:**
- Training data: First 2.5 years (Month 1 to 30)
- Test data: Last 6 months (Month 31 to 36)

**Why NOT random split:**
Imagine studying for an exam using questions from the future — 
that's cheating! A random split lets the model "see" future 
data during training, which is called **data leakage**. 
The model looks great in testing but fails in real life.

**Metrics to Measure Model Performance:**

| Metric | Simple Meaning |
|--------|---------------|
| **RMSE** | Average error in items_sold (big errors are penalised more) |
| **MAE** | Average error in items_sold (easy to understand) |
| **R² Score** | How well the model explains the data (closer to 1 = better) |
| **Ranking Accuracy** | How often the model picks the right promotion |

---

### B3(b) — Why Different Recommendations for Same Store?

**Example:** Model says Loyalty Points for Store 12 in December,
but Flat Discount for Store 12 in March.

**How to investigate using Feature Importance:**

Step 1 — Check global feature importance from Random Forest:
- It tells us which features matter most overall 
  (e.g., month, is_festival, competition_density)

Step 2 — Use SHAP values for Store 12 specifically:
- In December: `is_festival=1` and `month=12` push 
  the model toward Loyalty Points 
  (festive shoppers prefer long-term rewards)
- In March: `is_festival=0`, lower footfall, and 
  high competition push toward Flat Discount 
  (customers want immediate savings)

**How to explain this to marketing team (non-technical):**
> "In December, customers are in festive mood and respond 
> better to loyalty rewards — they plan to shop again. 
> In March, there is more competition nearby and customers 
> prefer instant discounts. The model learned this pattern 
> from 3 years of historical data."

---

### B3(c) — How to Deploy and Monitor the Model

**Step 1 — Save the Model After Training:**
```python
import joblib
joblib.dump(pipeline, 'promotion_model_v1.pkl')
```
This saves the entire model including all preprocessing steps.

**Step 2 — Every Month, Prepare New Data:**
- Collect store data, upcoming promotions, and calendar info
- Apply same feature engineering as during training
- Load saved model: `model = joblib.load('promotion_model_v1.pkl')`

**Step 3 — Generate Recommendations:**
- For each store, create 5 rows (one per promotion)
- Run all through the model
- Pick the promotion with highest predicted items_sold
- Send recommendation report to marketing team

**Step 4 — Monitor If Model Is Still Working:**

| What to Check | How Often | What to Do If Problem Found |
|--------------|-----------|----------------------------|
| Compare predicted vs actual items_sold | Every month | If error increases a lot, retrain |
| Check if data looks different than training data | Every month | If yes, retrain the model |
| Check if recommended promotions are actually working | Every month | If not, investigate and fix |
| Full retraining with all new data | Every 6 months | Always retrain before festive season |

**Version your model** (v1, v2, v3...) so you can go back 
to an older version if a new one performs worse.