## 2026-03-28 - Highlight primary path for Train/Re-Train button
**Learning:** Adding the Streamlit parameter `type="primary"` to the "Train / Re-train Model" button significantly draws user attention to the main required action, whereas default button styles blend into the configuration sidebar and can cause hesitation on what to do next.
**Action:** Use `type="primary"` to explicitly anchor the main interactive element on any form or sidebar that initiates the core compute path.

## 2026-03-31 - Preemptively disable dependent buttons
**Learning:** Instead of relying on post-click error messages when a prerequisite isn't met (e.g., trying to train a model without selecting features, or trying to fine-tune without an existing model), preemptively disabling the button with a `help` tooltip explaining the requirement provides a much better, less frustrating user experience and prevents wasted interactions.
**Action:** Use the `disabled` parameter along with a `help` tooltip on Streamlit buttons to explicitly communicate prerequisites before the user clicks.
