import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Contoh data
x = np.linspace(0, 10, 100)
df = pd.DataFrame({
    'x': x,
    'sin(x)': np.sin(x),
    'cos(x)': np.cos(x),
})

# Reshape ke long format untuk Altair
df_long = df.melt('x', var_name='Fungsi', value_name='Nilai')

# Plot interaktif
chart = alt.Chart(df_long).mark_line().encode(
    x=alt.X('x', title='Waktu (s)'),
    y=alt.Y('Nilai', title='Amplitudo'),
    color='Fungsi',
    tooltip=['x', 'Fungsi', 'Nilai']
).properties(
    title='Perbandingan sin(x) dan cos(x)',
    width=700,
    height=400
).interactive()  # <- untuk zoom dan pan

# Tampilkan di Streamlit
st.altair_chart(chart, use_container_width=True)
