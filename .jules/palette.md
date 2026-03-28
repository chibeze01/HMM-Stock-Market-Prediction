## 2026-03-28 - Highlight primary path for Train/Re-Train button
**Learning:** Adding the Streamlit parameter `type="primary"` to the "Train / Re-train Model" button significantly draws user attention to the main required action, whereas default button styles blend into the configuration sidebar and can cause hesitation on what to do next.
**Action:** Use `type="primary"` to explicitly anchor the main interactive element on any form or sidebar that initiates the core compute path.
