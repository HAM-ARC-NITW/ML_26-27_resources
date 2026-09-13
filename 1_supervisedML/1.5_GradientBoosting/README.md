# Gradient Boosting

---

*HAM ARC ML Sessions, 2026–27.*

Gradient Boosting is a supervised learning technique that combines multiple **decision trees** to build a strong predictive model.

Instead of trying to build one complicated tree, Gradient Boosting builds trees **sequentially**, where every new tree tries to correct the mistakes made by the previous model.

It can be used for both:

- **Regression** — predicting a continuous value
- **Classification** — predicting a class or probability

---

## 1. What is it?

Gradient Boosting is an **ensemble learning algorithm** in which multiple decision trees are added one after another.

The key idea is:

> **Each new tree learns from the errors of the previous model.**

For example, suppose we want to predict house prices.

The first model might make predictions that are reasonably close, but not perfect:

```text
Actual:      200   250   300   350
Prediction:  180   270   280   320
