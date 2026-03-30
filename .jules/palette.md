## 2026-03-28 - Highlight primary path for Train/Re-Train button
**Learning:** Adding the Streamlit parameter `type="primary"` to the "Train / Re-train Model" button significantly draws user attention to the main required action, whereas default button styles blend into the configuration sidebar and can cause hesitation on what to do next.
**Action:** Use `type="primary"` to explicitly anchor the main interactive element on any form or sidebar that initiates the core compute path.

## 2026-04-03 - Prevent Error States with Proactive Disabling
**Learning:** Post-click error messages (e.g., clicking "Fine-Tune" before "Train") create a disjointed experience. Preemptively disabling dependent Streamlit buttons using the `disabled` parameter and a `help` tooltip provides immediate clarity on prerequisites and prevents user frustration.
**Action:** Always use `disabled` paired with a `help` string for buttons that require a previous state or action to be completed first, rather than relying on error messages after the user has attempted the action.
