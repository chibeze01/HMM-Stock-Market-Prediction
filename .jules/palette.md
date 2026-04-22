## 2026-03-29 - Enhance Visual Hierarchy in Streamlit Sidebars
**Learning:** In form-heavy Streamlit apps, it's critical to use `type="primary"` on the main call-to-action buttons (like "Train Model" or "Submit") to visually distinguish them from secondary actions like "Reset" or input elements. This explicitly anchors the main interactive element for users and improves cognitive flow.
**Action:** Always explicitly set `type="primary"` for the primary submission or progression action in any multi-input form or sidebar.
## 2026-03-28 - Highlight primary path for Train/Re-Train button
**Learning:** Adding the Streamlit parameter `type="primary"` to the "Train / Re-train Model" button significantly draws user attention to the main required action, whereas default button styles blend into the configuration sidebar and can cause hesitation on what to do next.
**Action:** Use `type="primary"` to explicitly anchor the main interactive element on any form or sidebar that initiates the core compute path.

## 2026-04-03 - Prevent Error States with Proactive Disabling
**Learning:** Post-click error messages (e.g., clicking "Fine-Tune" before "Train") create a disjointed experience. Preemptively disabling dependent Streamlit buttons using the `disabled` parameter and a `help` tooltip provides immediate clarity on prerequisites and prevents user frustration.
**Action:** Always use `disabled` paired with a `help` string for buttons that require a previous state or action to be completed first, rather than relying on error messages after the user has attempted the action.
## 2026-03-31 - Preemptively disable dependent buttons
**Learning:** Instead of relying on post-click error messages when a prerequisite isn't met (e.g., trying to train a model without selecting features, or trying to fine-tune without an existing model), preemptively disabling the button with a `help` tooltip explaining the requirement provides a much better, less frustrating user experience and prevents wasted interactions.
**Action:** Use the `disabled` parameter along with a `help` tooltip on Streamlit buttons to explicitly communicate prerequisites before the user clicks.
## 2026-04-01 - Preemptively disable fine-tune button and fix stale UI state
**Learning:** In Streamlit apps, dependent UI elements (like buttons that require an action to be completed first) can appear in a 'stale' state after a state-mutating action is performed because the script execution stops rendering elements that have already been evaluated. Also, disabling an action visually and explaining why with a tooltip provides better UX than allowing the action and showing an error afterward.
**Action:** When a main action (like model training) updates the session state in a way that unlocks other UI components, add `st.rerun()` at the end of the success block to force an immediate re-render and fix stale UI states. Additionally, preemptively disable dependent buttons using `disabled` and `help` tooltips to explain prerequisites.

## 2026-04-12 - Improve empty state with step-by-step Getting Started guide
**Learning:** Providing a numbered 'Getting Started' step-by-step guide in empty states (e.g., when a model hasn't been trained yet) significantly reduces cognitive load and orients users compared to a generic instruction.
**Action:** Enhance empty states by explicitly guiding users through the required actions with numbered lists or similar clear visual structures.

## 2026-04-22 - Enhance Empty States with Structured Guidance
**Learning:** Generic, text-only empty states (e.g., "Not enough data") leave users guessing about how to resolve the issue. By structuring empty states with bold headers, providing clear, actionable next steps (like "Try expanding the date range"), and including contextual icons, you significantly reduce cognitive load and improve the user's sense of control.
**Action:** Always format empty state messages using structured text (bold headers), actionable guidance, and visual cues (icons) rather than generic single-line strings.
