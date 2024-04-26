import streamlit as st
import numpy as np
import tensorflow as tf
from joblib import load
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Importing the load_data function from the model_training module
from model_training import load_data

# Load the model and the preprocessor
@st.cache_resource  # Cache resources like model, preprocessor, and data
def load_resources():
    model = tf.keras.models.load_model('hr_recruitment_model.h5')
    preprocessor = load('preprocessor.joblib')  # Load the saved preprocessor
    data = load_data('data/HR Dataset.xlsx')  # Load the existing employee data
    return model, preprocessor, data

model, preprocessor, existing_employees_data = load_resources()

st.title('HR Recruitment Performance Predictor')
st.markdown("""
Welcome to the HR Recruitment Performance Predictor! This sophisticated AI model takes a comprehensive look at various factors to predict the performance category of potential hires and compare their profiles against existing employees.

**Factors Considered for Prediction:**
- **Demographic and Background**: Age, Gender, Education Level
- **Professional Experience**: Years of Working Experience, Years of Industry Experience
- **Position-Related**: Role Level, Previous Company Tier, Department
- **Personality and Origin**: Personality Type, Origin

The model evaluates potential hires by considering a comprehensive set of factors including demographic information, professional experience, and personality traits. The predictive analysis focuses on:
- **Performance Scores**: Predicted average performance scores are calculated based on the individual's role, experience, and background, reflecting their expected contribution to the company.
- **Promotional Prospects**: The likelihood and frequency of promotions the candidate might receive, which indicate their potential for growth and success within the organization.

These outcomes help in assessing the overall value a potential hire might bring to the company and guide decision-making in the recruitment process.

**How to Use This Tool:**
1. **Enter Employee Details**: Provide the necessary details about the employee in the input form.
2. **Submit for Prediction**: After entering the details, click the 'Predict' button to process the data and generate the performance category.
3. **Review the Results**: Examine the prediction outcome and attribute comparison to make informed recruitment decisions.
""")

# Input form
with st.form("employee_input_form"):
    st.header("Enter the employee details:")
    age = st.slider('Age', min_value=18, max_value=65, value=30)
    gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
    education_level = st.selectbox('Education Level', ['High School', 'Bachelor', 'Master', 'PhD'])
    working_experience = st.slider('Years of Working Experience', min_value=0, max_value=47, value=5)
    industry_experience = st.slider('Years of Industry Experience', min_value=0, max_value=47, value=3)
    role_level = st.selectbox('Role Level', ['Entry', 'Mid', 'Senior', 'Management'])
    personality_type = st.selectbox('Personality Type', ['INTJ', 'ENTP', 'INFP', 'ENFJ', 'ISTJ', 'ISFJ', 'INFJ', 'INTP', 'ESTP',
                                                        'ESFP', 'ENFP', 'ESTJ', 'ESFJ', 'ISTP', 'ISFP', 'ENTJ'])
    origin = st.selectbox('Origin', ['Local', 'Foreigner'])
    previous_company_tier = st.selectbox('Previous Company Tier', ['Tier 1', 'Tier 2', 'Tier 3'])
    department = st.selectbox('Department', ['Sales', 'Marketing', 'Development', 'Human Resources'])
    referral = st.radio('Referral', [0, 1])

    submitted = st.form_submit_button("Predict")
    if submitted:
        input_data = pd.DataFrame([[
            age, gender, education_level, working_experience, industry_experience, role_level,
            personality_type, origin, previous_company_tier, department, referral
        ]], columns=[
            'Age', 'Gender', 'Education Level', 'Years of Working Experience', 'Years of Industry Experience', 'Role Level',
            'Personality Type', 'Origin', 'Previous Company Tier', 'Department', 'Referral'
        ])
        X_processed = preprocessor.transform(input_data)
        prediction = model.predict(X_processed)[0]

        # Find the index of the highest predicted category
        predicted_category_index = np.argmax(prediction)
        predicted_category = ['Underperform', 'Normal', 'Overperform'][predicted_category_index]

        # Get the percentage of the highest predicted category
        predicted_percentage = prediction[predicted_category_index] * 100

        # Description based on the predicted category
        if predicted_category == 'Underperform':
            description = f"""
            **Predicted Performance Category: Underperform**
            
            The model predicts that this candidate is likely to underperform relative to the role expectations. The predicted probability for this category is {predicted_percentage:.2f}%. This may be due to a mismatch in experience levels, skill sets, or possibly the personality type not aligning well with the department's culture. Consider whether additional training or a different role might be more suitable for this candidate.
            """
        elif predicted_category == 'Normal':
            description = f"""
            **Predicted Performance Category: Normal**
            
            The prediction suggests that this candidate is likely to meet the expected performance standards for the role. The predicted probability for this category is {predicted_percentage:.2f}%. They appear to have a suitable match in terms of experience and skills that align well with what is typically expected in this position. This candidate should be considered a solid potential hire.
            """
        elif predicted_category == 'Overperform':
            description = f"""
            **Predicted Performance Category: Overperform**
            
            This candidate shows potential to exceed the typical performance expectations for the role. The predicted probability for this category is {predicted_percentage:.2f}%. Their profile suggests a strong alignment with the required qualifications and an ability to bring additional value to the team. They are highly recommended for roles that can leverage their full potential and provide room for growth.
            """

        st.markdown(description)

        # Compare to average values from existing employee data for numerical columns
        cols_to_compare = ['Age', 'Years of Working Experience', 'Years of Industry Experience']
        average_data = existing_employees_data[cols_to_compare].mean()
        candidate_data = input_data[cols_to_compare].iloc[0]

        # Create DataFrame for Plotly chart
        comparison_data = pd.DataFrame({
            'Attribute': cols_to_compare,
            'Candidate': candidate_data.values,
            'Average Employee': average_data.values
        })

        # Plot using Plotly
        fig = px.bar(comparison_data, x='Attribute', y=['Candidate', 'Average Employee'],
                     barmode='group', title="Candidate vs Average Employee Comparison")
        st.plotly_chart(fig)

        with st.expander("See detailed analysis of Candidate vs. Average Employee"):
            st.markdown("""
            #### Candidate vs. Average Employee Analysis
            - **Overview**: This section compares the candidate's key attributes—age, working experience, and industry experience—with the averages of existing employees to evaluate alignment and uncover potential strengths or development areas.
            - **Attributes Compared**:
                - **Age**: Reflects on the candidate's maturity and potential cultural fit within the team.
                - **Years of Working Experience**: Assesses the breadth of the candidate's professional background and readiness for the role.
                - **Years of Industry Experience**: Evaluates the depth of the candidate's knowledge and skills specific to the industry.
            """)


            # Initialize the markdown string
            markdown_text = "- **Key Observations**:\n"

            # Dynamically generate observations
            for col in cols_to_compare:
                diff = candidate_data[col] - average_data[col]
                abs_diff = abs(diff)  # Use the absolute value to avoid negative numbers in the text
                if col == 'Age':
                    if diff > 0:
                        markdown_text += f"    - **Age**: At **{abs_diff:.2f} years older** than the average, the candidate may bring a higher level of maturity and potentially more conservative values to the team, which can be beneficial in roles requiring prudent decision-making.\n"
                    elif diff < 0:
                        markdown_text += f"    - **Age**: Being **{abs_diff:.2f} years younger** suggests the candidate could inject fresh energy and perhaps a more modern approach to teamwork and innovation.\n"
                    else:
                        markdown_text += "    - **Age**: Matching the average age suggests the candidate is likely to integrate well with the team's current demographic dynamics.\n"
                elif col == 'Years of Working Experience':
                    if diff > 0:
                        markdown_text += f"    - **Working Experience**: With **{abs_diff:.2f} more years** of experience, the candidate is likely to possess advanced skills and insights, which could translate into immediate effectiveness and leadership potential within the role.\n"
                    elif diff < 0:
                        markdown_text += f"    - **Working Experience**: Having **{abs_diff:.2f} fewer years** of experience may indicate the candidate is still developing their skills, which presents an opportunity for growth and tailored training within your organization.\n"
                    else:
                        markdown_text += "    - **Working Experience**: Being on par with the average suggests the candidate has sufficient experience to meet job expectations without significant additional training.\n"
                elif col == 'Years of Industry Experience':
                    if diff > 0:
                        markdown_text += f"    - **Industry Experience**: **{abs_diff:.2f} more years** of industry-specific experience suggests the candidate could bring valuable specialized knowledge, potentially filling critical knowledge gaps or enhancing team capabilities.\n"
                    elif diff < 0:
                        markdown_text += f"    - **Industry Experience**: **{abs_diff:.2f} fewer years** indicates that while the candidate may have relevant skills, they might need industry-specific training to fully align with role expectations and company practices.\n"
                    else:
                        markdown_text += "    - **Industry Experience**: Equivalence in industry experience implies the candidate should integrate smoothly with current operational standards and practices.\n"

            # Display the markdown text
            st.markdown(markdown_text)

            st.markdown("""
            - **Business Insights**:
                - **Strategic Alignment**: Evaluate how the candidate's attributes align with your company's strategic needs, particularly in filling leadership roles or areas needing innovation.
                - **Cultural Integration**: Consider the potential cultural impact, particularly with age and experience differences, to enhance team diversity or maintain stability.
                - **Training and Development**: Identify specific training programs that could be beneficial based on the candidate's existing experience relative to company norms.
                - **Succession Planning**: Assess potential for filling critical roles in the future, especially if the candidate brings significant industry or work experience.
            """)



        # Merge candidate's input data with the departmental personality counts
        input_department_data = pd.concat([existing_employees_data, input_data], ignore_index=True)

        # Group merged data by department and personality type
        input_department_personality_counts = input_department_data.groupby(['Department', 'Personality Type']).size().unstack(fill_value=0)

        # Create a Plotly heatmap figure for departmental distribution
        fig_department = go.Figure(data=go.Heatmap(
            z=input_department_personality_counts.values,
            x=input_department_personality_counts.columns,
            y=input_department_personality_counts.index,
            colorscale='Blues',
            hoverongaps=False))

        # Customize layout for departmental distribution
        fig_department.update_layout(
            title="Distribution of Personality Types by Department",
            xaxis_title="Personality Type",
            yaxis_title="Department",
            xaxis=dict(tickangle=-45),
        )

        st.plotly_chart(fig_department)

        with st.expander("See detailed analysis of MBTI type distribution"):
            st.markdown("""
            #### MBTI Type Distribution Analysis
            - **Overview**: This analysis examines the prevalence of the candidate's MBTI personality type within their chosen department and across the entire company.
            - **Context**:
                - **MBTI in Department**: Looks at how common the candidate's personality type is within the department, which can influence team dynamics and role suitability.
                - **MBTI in Company**: Considers the overall distribution of the personality type across the company, identifying potential areas for increased diversity or commonality.
     
            """)

            total_employees_in_department = input_department_personality_counts.sum(axis=1)
            percentage_candidate_mbti_department = (input_department_personality_counts.loc[department, personality_type] / total_employees_in_department[department] * 100)
            total_employees_in_company = input_department_personality_counts.sum().sum()
            percentage_candidate_mbti_company = (input_department_personality_counts[personality_type].sum() / total_employees_in_company * 100)

            department_observation = f"- **Departmental Distribution**: In the **{department}** department, the candidate's MBTI type ({personality_type}) is **{percentage_candidate_mbti_department:.2f}%** of the workforce."
            company_observation = f"- **Company-wide Distribution**: Across the entire company, the candidate's MBTI type ({personality_type}) makes up **{percentage_candidate_mbti_company:.2f}%** of employees."

            st.markdown(f"""      
            - **Key Observations**:
                - **Departmental Distribution**: In the **{department}** department, the candidate's MBTI type ({personality_type}) is **{percentage_candidate_mbti_department:.2f}%** of the workforce.
                - **Company-wide Distribution**: Across the entire company, the candidate's MBTI type ({personality_type}) makes up **{percentage_candidate_mbti_company:.2f}%** of employees.  
            """)


            st.markdown("""
            - **Business Insights**:
                - **Team Integration**: Understanding the commonality of the candidate's MBTI type within their department can help anticipate their integration challenges or ease. A rare type might bring diverse perspectives, whereas a common type might integrate more seamlessly.
                - **Cultural Fit**: Evaluate how the candidate's personality type aligns with the prevailing corporate culture and values, especially in departments where their type is underrepresented or overrepresented.
                - **Diversity Enhancement**: Consider the candidate's potential to contribute to personality diversity, which can enrich problem-solving and innovation within teams.
                - **Training and Development Needs**: Tailor personal development programs to ensure diverse personality types are well-supported in their roles, fostering a balanced and inclusive workplace environment.
            """)


        # Calculate the average profile of existing employees for the relevant attributes
        avg_profile = existing_employees_data[cols_to_compare].mean()

        # Create data for the Radar Chart
        categories = cols_to_compare
        avg_values = avg_profile.tolist()
        candidate_values = candidate_data.tolist()


        # Calculate the min and max for each attribute for normalization
        min_attributes = existing_employees_data[cols_to_compare].min()
        max_attributes = existing_employees_data[cols_to_compare].max()

        # Normalize the attributes of the candidate and the average profile
        normalized_avg_values = [(val - min_attributes[col]) / (max_attributes[col] - min_attributes[col]) if max_attributes[col] != min_attributes[col] else 0.5 for col, val in zip(cols_to_compare, avg_values)]
        normalized_candidate_values = [(val - min_attributes[col]) / (max_attributes[col] - min_attributes[col]) if max_attributes[col] != min_attributes[col] else 0.5 for col, val in zip(cols_to_compare, candidate_values)]


        # Create the radar chart
        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=normalized_candidate_values,
            theta=categories,
            fill='toself',
            name='Candidate Profile'
        ))

        fig.add_trace(go.Scatterpolar(
            r=normalized_avg_values,
            theta=categories,
            fill='toself',
            name='Average Employee Profile'
        ))

        # Update layout for better visualization
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            title="Normalized Radar Chart of Candidate vs Average Employee Profiles",
            showlegend=True
        )

        # Display the radar chart before the expander
        st.plotly_chart(fig)

        # Expander for detailed radar chart analysis
        with st.expander("Understand the Radar Chart"):
            st.markdown("""
            #### Candidate vs. Average Employee Radar Chart Analysis
            - **Overview**: This radar chart provides a visual comparison of the candidate's attributes to the average attributes of existing employees, normalized to a common scale.
            - **Usage**: Each axis represents a different attribute, and the distance from the center indicates how the candidate's attribute value compares to the normalized range of existing employees.
            - **Interpreting the Chart**:
                - A point on the outer edge of the chart indicates the candidate's attribute is at the maximum observed value amongst existing employees.
                - A point at the very center would indicate the attribute is at the minimum.
                - This normalization allows for direct comparison across attributes with different units and scales.
            """)


