## 2026-03-29 - Enhance Visual Hierarchy in Streamlit Sidebars
**Learning:** In form-heavy Streamlit apps, it's critical to use `type="primary"` on the main call-to-action buttons (like "Train Model" or "Submit") to visually distinguish them from secondary actions like "Reset" or input elements. This explicitly anchors the main interactive element for users and improves cognitive flow.
**Action:** Always explicitly set `type="primary"` for the primary submission or progression action in any multi-input form or sidebar.
