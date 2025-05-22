import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.signal import find_peaks

st.title('🔆 Glucosefitter.io 🔆')
st.write("Find the curve of best fit for a sample of glucose levels.")

def polynomial(x, *coeffs):
    return sum(c * x**i for i, c in enumerate(coeffs))

def linear(x, a, b):
    return a * x + b

def gaussian(x, a, b, c):
    return a * np.exp(-(x - b)**2 / (2 * c**2))

def logistic(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))

def arrhenius(x, a, b):
    return a * np.exp(-b / x)

def fit_curve(x_data, y_data, func, p0):
    try:
        popt, pcov = curve_fit(func, x_data, y_data, p0=p0)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except Exception as e:
        st.error('Oh darn, we cannot fit this graph.')
        return None, None

def calculate_error(x_data, y_data, fit_func, fit_params):
    residuals = y_data - fit_func(x_data, *fit_params)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    r_squared = 1 - (ss_res / ss_tot)
    average_error = np.mean(np.abs(residuals))
    return r_squared, average_error

def lighten_color(color, amount=0.5):
    try:
        c = mcolors.cnames[color]
    except KeyError:
        c = color
    c = mcolors.to_rgba(c)
    return [(1.0 - amount) * (1.0 - comp) + amount * comp for comp in c]

st.sidebar.header('Customize Your Graph')
line_color = st.sidebar.color_picker('Line Colour', '#1150FF')  # default to blue
point_color = st.sidebar.color_picker('Point Color', '#1418B7') # default to blue

def plot_fit(x_data, y_data, fit_func, fit_params, fit_label, title, x_label, y_label, line_color, point_color, errors=None):
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, label='Data', color=point_color)
    x_fit = np.linspace(min(x_data), max(x_data), 1000)
    y_fit = fit_func(x_fit, *fit_params)
    
    light_color = lighten_color(line_color, amount=0.7)
    
    if errors is not None:
        y_fit_lower = fit_func(x_fit, *(fit_params - errors))
        y_fit_upper = fit_func(x_fit, *(fit_params + errors))
        plt.fill_between(x_fit, y_fit_lower, y_fit_upper, color=light_color, alpha=0.2, label='Error margin')

    plt.plot(x_fit, y_fit, label=fit_label, color=line_color)
    plt.title("Curve of Best Fit")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()

    params_text = "\n".join([f"{param:.2f}" for param in fit_params])
    plt.annotate(f"Parameters:\n{params_text}", xy=(0.05, 0.95), xycoords='axes fraction',
                 verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

    st.pyplot(plt)

def create_table():
    data = pd.DataFrame(
        [[0, 0]] * 3,
        columns=['X', 'Y']
    )
    edited_data = st.data_editor(data, num_rows="dynamic", key='editable_table')
    return edited_data

st.sidebar.write('')
st.sidebar.header('Plug In Data')
data_input_option = st.sidebar.radio('Choose Method', ('Enter Manually', 'Upload CSV', 'Input Table'))

valid_data = False
equation_text = ""

if data_input_option == 'Enter Manually':
    x_values = st.sidebar.text_area('Enter X values', '1,2,3,4,5')
    y_values = st.sidebar.text_area('Enter Y values', '1,4,9,16,25')
    
    try:
        x_data = np.array([float(x) for x in x_values.split(',')])
        y_data = np.array([float(y) for y in y_values.split(',')])
        if len(x_data) != len(y_data):
            st.error('Oh jeez, amount of X and Y values must be equal please.')
        else:
            valid_data = True
    except ValueError:
        st.error('Hey! Numeric values for X and Y only please. Or maybe you put an extra comma.')
elif data_input_option == 'Upload CSV':
    uploaded_file = st.sidebar.file_uploader('Choose a CSV file', type='csv')
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file, header=None)
            if data.shape[1] != 2:
                st.error('Wow buddy, only two columns in the CSV please.')
            else:
                x_data = data[0].values
                y_data = data[1].values
                if not np.issubdtype(x_data.dtype, np.number) or not np.issubdtype(y_data.dtype, np.number):
                    st.error('Hold up, pal. Csv must have only numeric values.')
                elif len(x_data) != len(y_data):
                    st.error('Kiddo, the number of X and Y values in the CSV must be equal.')
                else:
                    st.write('Uploaded Data')
                    st.write(data)
                    valid_data = True
        except Exception as e:
            st.error(f'Error reading CSV file: {e}')
elif data_input_option == 'Input Table':
    st.subheader('Input Data Table')
    table_data = create_table()
    st.write("This table is all yours to edit.")

    x_data = table_data['X'].to_numpy()
    y_data = table_data['Y'].to_numpy()
    valid_data = True

st.subheader("Select a curve type to fit")
curve_type = st.selectbox("I'd like to graph...", ['Linear', 'Polynomial', 'Gaussian', 'Logistic', 'Arrhenius', 'Histogram'])

if valid_data:
    if curve_type == 'Polynomial':
        order = st.slider('Select polynomial order', 1, 10, 2)

        if order == 1:
            if np.all(y_data == y_data[0]):
                fit_func = lambda x, a, b: y_data[0] 
                p0 = [0, y_data[0]] 
                fit_label = f"y = {y_data[0]}"
            else:
                fit_func = linear
                p0 = [1, 0]
                fit_label = 'Linear fit: y = mx + b'
        elif np.all(y_data == y_data[0]):
            fit_func = lambda x, *coeffs: np.full_like(x, y_data[0])
            p0 = [y_data[0]]
            fit_label = f"y = {y_data[0]}"
        else:
            if order >= len(x_data):
                order = len(x_data) - 1
            fit_func = lambda x, *coeffs: polynomial(x, *coeffs)
            p0 = [1] * (order + 1)
            fit_label = f'Polynomial of order {order}'


    elif curve_type == 'Linear':

        if np.all(y_data == y_data[0]):
            fit_func = lambda x, a, b: y_data[0]  
            p0 = [0, y_data[0]]  
            fit_label = f"y = {y_data[0]}"
        else:
            fit_func = linear
            p0 = [1, 1]
            fit_label = 'Linear'

    elif curve_type == 'Gaussian':
        st.write('Note: To fit a Gaussian over frequency, use Histogram.')
        fit_func = gaussian
        p0 = [1, 0, 1]
        fit_label = 'Gaussian'

    elif curve_type == 'Logistic': 
        fit_func = logistic 
        p0 = [1, 1, 1] 
        fit_label = 'Logistic'

    elif curve_type == 'Arrhenius':
        fit_func = arrhenius
        p0 = [1, 1]
        fit_label = 'Arrhenius'
    
    elif curve_type == 'Histogram':
        
        st.write('')
        st.write('Please paste several data points below. Curve may not function in extreme cases.')

        x_values_input = st.text_area('Enter values here:', '81, 74, 67, 89, 72, 78, 84, 69, 76, 62, 81, 74, 67, 89, 72, 78, 84, 69, 76, 62, 80, 73, 88, 65, 92, 79, 85, 71, 63')
        try:
            x_data = np.array([float(x) for x in x_values_input.split(',')])
            valid_data = True
        except ValueError:
            st.error('Please enter valid numeric values for X.')
            x_data = np.array([])
            valid_data = False

        if valid_data and len(x_data) > 0:
            num_bins = min(10, len(x_data))
            fig, ax = plt.subplots()
            
            counts, bins, _ = ax.hist(x_data, bins=num_bins, alpha=0.7, color=point_color, edgecolor='white')
            
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            try:
                popt, _ = curve_fit(gaussian, bin_centers, counts, p0=[max(counts), np.mean(x_data), np.std(x_data)], maxfev=2000)
            except RuntimeError:
                st.error('Hmm. The Gaussian curve does not seem to work for this data set.')
                popt = None

            if popt is not None:
                x_fit = np.linspace(min(x_data), max(x_data), 100000)
                y_fit = gaussian(x_fit, *popt)
                ax.plot(x_fit, y_fit, label='Gaussian Fit', color=line_color)

            ax.set_title('Histogram with Gaussian Fit')
            ax.set_xlabel('Time')
            ax.set_ylabel('Frequency')
            ax.legend()
            st.pyplot(fig)
            fit_params = None
 
        
    if curve_type != 'Histogram':
        fit_params, fit_errors = fit_curve(x_data, y_data, fit_func, p0)

        if fit_params is not None:
            
            r_squared, avg_error = calculate_error(x_data, y_data, fit_func, fit_params)

            if curve_type == 'Polynomial':
                coeffs = fit_params
                terms = []

                for i, coeff in enumerate(coeffs[::-1]):
                    if not np.isclose(coeff, 0):
                        if i == 0:
                            term = f"{coeff:.2f}"
                        elif i == 1:
                            term = f"{coeff:.2f}x"
                        else:
                            term = f"{coeff:.2f}x^{i}"
                        terms.append(term)

                if np.all(y_data == 0):
                    equation_text = "y = 0"
                elif len(terms) == 1 and "x" not in terms[0]:
                    equation_text = f"y = {terms[0]}"
                elif len(terms) == 1 and terms[0] == "1.00x":  
                    equation_text = "y = x"
                elif not terms:
                    equation_text = "y = 0"
                else:
                    equation_text = "y = "
                    for term in terms:
                        coefficient = float(term.split('x')[0])
                        if coefficient > 0:
                            equation_text += f"+ {term} "
                        else:
                            equation_text += f"- {term.lstrip('+-')} "
                    equation_text = equation_text.replace("+ -", "- ").strip()

            elif curve_type == 'Linear':
                equation_text = f"y = {fit_params[0]:.2f}x {'+ ' if fit_params[1] >= 0 else '- '}{abs(fit_params[1]):.2f}"
            elif curve_type == 'Gaussian':
                equation_text = f"y = {fit_params[0]:.2f} e^{{-\\frac{{(x - {fit_params[1]:.2f})^2}}{{2 \\cdot {fit_params[2]:.2f}^2}}}}"
            elif curve_type == 'Logistic': 
                equation_text = f"y = \\frac{{{fit_params[0]:.2f}}}{{1 + e^{{-{fit_params[1]:.2f} \\cdot (x - {fit_params[2]:.2f})}}}}"
            elif curve_type == 'Arrhenius':
                equation_text = f"y = {fit_params[0]:.2f} e^{{-\\frac{{{fit_params[1]:.2f}}}{{x}}}}"

            st.latex(equation_text)
            plot_fit(x_data, y_data, fit_func, fit_params, fit_label, f"{fit_label} Curve Fit", "X values", "Y values", line_color, point_color, errors=fit_errors)

            if r_squared < 0.5: 
                st.write("This curve probably isn't the best choice for your data :/")
            with st.expander("""What's this "parameter" and "error" stuff?"""):
                st.write(f'Fitting parameters: {fit_params}')
                st.write(f'Fitting errors: {fit_errors}')
                st.write(f'R-squared: {r_squared:.2f}')
                st.write(f'Average error: {avg_error:.2f}')
                st.write("""
                    - **Fitting Parameters:** coefficients that define the shape and position of the fitted curve.
                    - **Fitting Errors:** represent the uncertainties or standard deviations of the fitted parameters. Inf means the value has essentially perfect fit.
                    - **R-squared:** a statistical measure of how well the regression predictions approximate the real data points. An R-squared value of 1 indicates a perfect fit.
                    - **Average Error:** represents the mean absolute error between the observed data points and the fitted curve.
                """)

        if valid_data == False:
            st.error('The program was unable to fit the curve. Please try a different function.')
