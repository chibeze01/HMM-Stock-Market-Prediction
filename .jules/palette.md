## 2026-03-28 - Highlight primary path for Train/Re-Train button
**Learning:** Adding the Streamlit parameter `type="primary"` to the "Train / Re-train Model" button significantly draws user attention to the main required action, whereas default button styles blend into the configuration sidebar and can cause hesitation on what to do next.
**Action:** Use `type="primary"` to explicitly anchor the main interactive element on any form or sidebar that initiates the core compute path.

## 2026-04-01 - Preemptively disable fine-tune button and fix stale UI state
**Learning:** In Streamlit apps, dependent UI elements (like buttons that require an action to be completed first) can appear in a 'stale' state after a state-mutating action is performed because the script execution stops rendering elements that have already been evaluated. Also, disabling an action visually and explaining why with a tooltip provides better UX than allowing the action and showing an error afterward.
**Action:** When a main action (like model training) updates the session state in a way that unlocks other UI components, add `st.rerun()` at the end of the success block to force an immediate re-render and fix stale UI states. Additionally, preemptively disable dependent buttons using `disabled` and `help` tooltips to explain prerequisites.
