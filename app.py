# ... (ALL YOUR ORIGINAL CODE REMAINS SAME ABOVE)

# Trigger Email Section
st.subheader("Trigger Email Actions (Only for Negative Reviews meeting threshold)")
negative_df = df[df["Email_Trigger"] == "Yes"].reset_index(drop=True)

for idx, row in negative_df.iterrows():
    uid = row.get('Unique_ID', f'Row {idx+1}')
    expander_key = f"expander_{idx}"

    expanded = st.session_state.open_expander_index == idx

    with st.expander(f"Email for Review #{idx+1} - {uid}", expanded=expanded):
        st.markdown(f"**Category:** {row.get('Category', 'N/A')}")
        st.markdown(f"**Date:** {row.get('Date', 'N/A')}")
        st.markdown(f"**Review:** {row['Review_text']}")
        st.markdown(f"**Response to be sent:** {row['Response']}")
        st.markdown(f"**Model Confidence:** {row['Confidence']:.2f} (threshold {NEGATIVE_THRESHOLD:.2f})")

        # ---------------------------
        # ✅ NEW: Manual Response Option
        # ---------------------------
        use_manual = st.checkbox(
            "Use Manual Response",
            key=f"use_manual_{idx}"
        )

        manual_response = ""
        if use_manual:
            manual_response = st.text_area(
                "Enter Manual Response",
                key=f"manual_text_{idx}",
                height=100
            )

        if st.button(f"Send Email (Row {idx})", key=f"send_button_{idx}"):
            recipient_email = row.get("Email", "")
            st.session_state.open_expander_index = idx

            # ---------------------------
            # ✅ NEW: Decide response
            # ---------------------------
            final_response = (
                manual_response.strip()
                if use_manual and manual_response.strip()
                else row["Response"]
            )

            if recipient_email:
                subject = f"Response to your review (ID: {uid})"
                body = (
                    f"Dear Customer,\n\n"
                    f"Thank you for your feedback. Please find our response below.\n\n"
                    f"---\n"
                    f"Review Details:\n"
                    f"ID: {uid}\n"
                    f"Category: {row.get('Category', 'N/A')}\n"
                    f"Date: {row.get('Date', 'N/A')}\n"
                    f"Review:\n{row['Review_text']}\n\n"
                    f"Our Response:\n{final_response}\n"
                    f"---\n\n"
                    f"Best regards,\nCustomer Support Team"
                )
                if send_email(recipient_email, subject, body):
                    st.success(f"Email sent to {recipient_email}")
            else:
                st.warning("No Email address found in this row.")

# ... (REST OF YOUR CODE REMAINS SAME)
