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

## 2026-04-06 - Replace generic empty states with actionable Getting Started guides
**Learning:** Presenting a generic informational message (like "Train the model to unlock evaluation") in an empty application state leaves users guessing about the exact steps required to reach the desired state. Providing a structured, numbered "Getting Started" guide significantly reduces cognitive load and provides a clear path to value.
**Action:** When designing an application's initial empty state, replace generic "do x to get y" messages with explicit, numbered, step-by-step instructions that map directly to the application's UI controls.
## 2026-04-12 - Improve empty state with step-by-step Getting Started guide
**Learning:** Providing a numbered 'Getting Started' step-by-step guide in empty states (e.g., when a model hasn't been trained yet) significantly reduces cognitive load and orients users compared to a generic instruction.
**Action:** Enhance empty states by explicitly guiding users through the required actions with numbered lists or similar clear visual structures.
## 2024-04-24 - Enhance Empty States for Component-level Missing Data
**Learning:** Generic informational messages (like `st.info("No data available")`) in empty states increase cognitive load and leave users confused about how to resolve the issue. Providing a clear header, actionable next steps, and a relevant icon makes the UI much more helpful and intuitive.
**Action:** Always enhance component-level empty states (e.g., when evaluation metrics are missing) by using structured text (bolded headers), explicitly describing the actionable next step (like expanding the date range), and including appropriate icons.

## 2026-04-25 - Enhance component-level empty states with visual hierarchy
**Learning:** Simple text messages like 'Not enough data' in Streamlit empty states lack visual impact and actionable guidance. Adding bolded headers, helpful next steps (e.g., 'Try extending the date range'), and icons (e.g., '📈') to `st.info` strings dramatically improves visual hierarchy, reduces cognitive load, and helps users recover from empty states faster.
**Action:** When implementing empty states for Streamlit components, always use structured formatting (bold headers), actionable guidance, and visual anchors (icons).
