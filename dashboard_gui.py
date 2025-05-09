import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
from pytz import timezone
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
from PIL import Image, ImageTk
import io

# --- Helper functions ---
def parse_size(x):
    if pd.isna(x): return np.nan
    x = x.strip()
    if x.endswith('M'): return float(x[:-1])
    if x.endswith('k'): return float(x[:-1]) / 1000
    return np.nan

def is_time_allowed(start_hour, end_hour, override=False):
    return True if override else start_hour <= datetime.now(timezone('Asia/Kolkata')).hour < end_hour

def plotly_fig_to_photo(fig):
    img_bytes = fig.to_image(format="png")
    img = Image.open(io.BytesIO(img_bytes))
    return ImageTk.PhotoImage(img)

# --- Load and preprocess data ---
apps_df = pd.read_csv('data/Play Store Data.csv')
reviews_df = pd.read_csv('data/User Reviews.csv')

apps_df = apps_df[apps_df['Category'] != '1.9']
apps_df['Installs'] = pd.to_numeric(apps_df['Installs'].str.replace('[+,]', '', regex=True), errors='coerce').fillna(0)
apps_df['Price'] = pd.to_numeric(apps_df['Price'].str.replace('$', ''), errors='coerce').fillna(0)
apps_df['Reviews'] = pd.to_numeric(apps_df['Reviews'], errors='coerce').fillna(0)
apps_df['SizeMB'] = apps_df['Size'].replace('Varies with device', np.nan).apply(parse_size)
apps_df['Android Ver'] = apps_df['Android Ver'].replace('Varies with device', np.nan)
apps_df['AndroidVerNumeric'] = pd.to_numeric(apps_df['Android Ver'].str.extract(r'(\d+\.\d+)')[0], errors='coerce').fillna(0)
apps_df['Last Updated'] = pd.to_datetime(apps_df['Last Updated'], errors='coerce')
apps_df['Revenue'] = apps_df['Installs'] * apps_df['Price']

# --- Dashboard Class ---
class PlayStoreDashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Play Store Data Dashboard")
        self.geometry("1200x900")
        self.view_all = False

        self.control_frame = tk.Frame(self)
        self.control_frame.pack(fill='x', pady=10)
        self.current_frame = tk.Frame(self)
        self.current_frame.pack(expand=1, fill='both')

        for i in range(1, 8):
            btn = tk.Button(self.control_frame, text=f"Run Task {i}", command=lambda n=i: self.run_single_task(n))
            btn.pack(side='left', padx=5)

        self.override_button = tk.Button(self.control_frame, text="Show All Tasks (Ignore Time)", command=self.show_all_tasks)
        self.override_button.pack(side="left", padx=10)

    def clear_current_frame(self):
        for widget in self.current_frame.winfo_children():
            widget.destroy()

    def run_task(self, task_number, frame, label):
        try:
            getattr(self, f"task{task_number}")(frame, override=self.view_all)
        except Exception as e:
            self.add_fallback(frame, f"Error: {str(e)}")
        label.destroy()

    def add_fallback(self, frame, message):
        tk.Label(frame, text=message, font=("Arial", 14)).pack(pady=20)

    def run_single_task(self, task_number):
        self.view_all = False
        self.clear_current_frame()
        frame = tk.LabelFrame(self.current_frame, text=f"Task {task_number}", padx=10, pady=10)
        frame.pack(fill='both', expand=True, padx=5, pady=5)
        label = tk.Label(frame, text="Loading, please wait...", font=("Arial", 12))
        label.pack(pady=5)
        self.run_task(task_number, frame, label)

    def show_all_tasks(self):
        self.view_all = True
        self.clear_current_frame()

        canvas = tk.Canvas(self.current_frame)
        scrollbar = tk.Scrollbar(self.current_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for task_number in range(1, 8):
            frame = tk.LabelFrame(scrollable_frame, text=f"Task {task_number}", padx=10, pady=10)
            frame.pack(fill='both', expand=True, padx=5, pady=5)
            label = tk.Label(frame, text="Loading, please wait...", font=("Arial", 12))
            label.pack(pady=5)
            self.run_task(task_number, frame, label)
    def task1(self, frame, override=False):
        print("Running Task 1 (Word Cloud)")

        health_apps = apps_df[apps_df['Category'] == 'HEALTH_AND_FITNESS'][['App', 'Rating']]
        merged = pd.merge(reviews_df, health_apps, on='App', how='inner')
        merged = merged[(merged['Rating'] == 5.0) & (merged['Translated_Review'].notna())]

        if merged.empty:
            self.add_fallback(frame, "No 5-star reviews found for Health & Fitness apps.")
            return

        text = ' '.join(merged['Translated_Review'].astype(str).tolist())

        stopwords = set(STOPWORDS)
        for app in merged['App'].unique():
            for word in app.lower().split():
                stopwords.add(word)

        wordcloud = WordCloud(width=800, height=400, stopwords=stopwords, background_color='white').generate(text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=1, fill='both')
        plt.close(fig)
        print("[DEBUG] Word cloud rendered.")

    def task2(self, frame, override=False):
        print("Running Task 2")

        if not is_time_allowed(13, 14, self.view_all):
            self.add_fallback(frame, "This visualization is only available between 1 PM and 2 PM IST.")
            return

        df = apps_df[
            (apps_df['Installs'] >= 10000) &
            (apps_df['Revenue'] >= 10000) &
            (apps_df['AndroidVerNumeric'] > 4.0) &
            (apps_df['SizeMB'] > 15) &
            (apps_df['Content Rating'] == 'Everyone') &
            (apps_df['App'].str.len() <= 30)
        ]

        top3 = df.groupby('Category')['Installs'].sum().nlargest(3).index
        df = df[df['Category'].isin(top3)]

        stats = df.groupby(['Category', 'Type']).agg(
            avg_installs=('Installs', 'mean'),
            avg_revenue=('Revenue', 'mean')
        ).reset_index()

        free = stats[stats['Type'] == 'Free'].set_index('Category').reindex(top3).fillna(0).infer_objects()
        paid = stats[stats['Type'] == 'Paid'].set_index('Category').reindex(top3).fillna(0)

        x = np.arange(len(top3))
        width = 0.35
        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.bar(x - width/2, free['avg_installs'], width, label='Free Avg Installs')
        ax1.bar(x + width/2, paid['avg_installs'], width, label='Paid Avg Installs')
        ax1.set_ylabel('Average Installs')
        ax1.set_xticks(x)
        ax1.set_xticklabels(top3, rotation=45)

        ax2 = ax1.twinx()
        ax2.plot(x - width/2, free['avg_revenue'], marker='o', color='green', label='Free Avg Revenue')
        ax2.plot(x + width/2, paid['avg_revenue'], marker='o', color='orange', label='Paid Avg Revenue')
        ax2.set_ylabel('Average Revenue')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=1, fill='both')
        plt.close(fig)
        print("Task 2 completed.")

    def task3(self, frame, override=False):
        print("Running Task 3 (Choropleth)...")

        current_hour = datetime.now(timezone('Asia/Kolkata')).hour
        if not self.view_all and not (18 <= current_hour < 20):
            self.add_fallback(frame, "This graph is only available between 6 PM and 8 PM IST.")
            return

        cat_installs = apps_df.groupby('Category')['Installs'].sum()
        top_categories = cat_installs.nlargest(5).index.tolist()
        filtered_cats = [cat for cat in top_categories if not cat.startswith(('A', 'C', 'G', 'S'))]
        top_data = apps_df[apps_df['Category'].isin(filtered_cats)].copy()
        if top_data.empty:
            self.add_fallback(frame, "No categories match criteria after filtering.")
            return

        top_data = top_data.sample(n=500) if len(top_data) > 500 else top_data

        import random
        country_list = ['United States', 'India', 'Brazil', 'Germany', 'Canada', 'UK', 'Australia', 'France', 'Mexico', 'Japan']
        top_data['Country'] = [random.choice(country_list) for _ in range(len(top_data))]

        summary = top_data.groupby(['Country', 'Category'])['Installs'].sum().reset_index()
        fig = px.choropleth(
            summary,
            locations='Country',
            locationmode='country names',
            color='Installs',
            hover_name='Category',
            color_continuous_scale='Viridis',
            title='Global Installs by Category (Top 5)'
        )
        fig.show()
        self.add_fallback(frame, "Interactive Choropleth map opened in browser.")

    def task4(self, frame, override=False):
        print("Running Task 4 (Grouped Bar Chart)...")
        if not is_time_allowed(15, 17, self.view_all):
            self.add_fallback(frame, "This visualization is only available between 3 PM and 5 PM IST.")
            return
        top10 = apps_df.groupby('Category')['Installs'].sum().nlargest(10).index
        valid = []
        for cat in top10:
            subset = apps_df[apps_df['Category'] == cat]
            if subset['Rating'].mean() >= 4.0 and subset['SizeMB'].mean() >= 10:
                if subset['Last Updated'].dt.month.eq(1).any():
                    valid.append(cat)
        if not valid:
            self.add_fallback(frame, "No categories match Task 4 filters.")
            return
        df = apps_df[apps_df['Category'].isin(valid)]
        avg_ratings = df.groupby('Category')['Rating'].mean().reindex(valid)
        total_reviews = df.groupby('Category')['Reviews'].sum().reindex(valid)
        x = np.arange(len(valid))
        width = 0.4
        fig, ax1 = plt.subplots(figsize=(10, 6))
        bar1 = ax1.bar(x - width/2, avg_ratings, width, label='Avg Rating', color='skyblue')
        ax1.set_ylabel('Average Rating')
        ax1.set_ylim(0, 5)
        ax2 = ax1.twinx()
        bar2 = ax2.bar(x + width/2, total_reviews, width, label='Total Reviews', color='orange')
        ax2.set_ylabel('Total Reviews')
        ax1.set_xticks(x)
        ax1.set_xticklabels(valid, rotation=45, ha='right')
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=1, fill='both')
        plt.close(fig)

    def task5(self, frame, override=False):
        print("Running Task 5 (Time Series Chart)")
        if not is_time_allowed(18, 21, self.view_all):
            self.add_fallback(frame, "This visualization is only available between 6 PM and 9 PM IST.")
            return
        df = apps_df.copy()
        df = df[(df['Content Rating'] == 'Teen') & df['App'].str.startswith('E') & (df['Installs'] > 10000)]
        df = df.dropna(subset=['Last Updated'])
        df['Month'] = df['Last Updated'].dt.to_period('M')
        pivot = df.groupby(['Month', 'Category'])['Installs'].sum().unstack().fillna(0)
        pivot.index = pivot.index.to_timestamp()
        fig, ax = plt.subplots(figsize=(10, 6))
        for col in pivot.columns:
            series = pivot[col]
            ax.plot(pivot.index, series, label=col)
            growth = series.pct_change()
            for i in range(1, len(growth)):
                if growth.iloc[i] > 0.2:
                    ax.fill_between([pivot.index[i-1], pivot.index[i]], [0, 0], [series.iloc[i-1], series.iloc[i]], color='lightgreen', alpha=0.3)
        ax.set_title('Monthly Installs Over Time by Category')
        ax.set_ylabel('Installs')
        ax.set_xlabel('Month')
        ax.legend()
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=1, fill='both')
        plt.close(fig)

    def task6(self, frame, override=False):
        print("Running Task 6 (Bubble Chart)")

        current_hour = datetime.now(timezone('Asia/Kolkata')).hour
        if not self.view_all and not (17 <= current_hour < 19):
            self.add_fallback(frame, "This graph is only available between 5 PM and 7 PM IST.")
            return

        df = apps_df.copy()
        df = df[(df['Rating'] > 3.5) & 
                (df['Installs'] > 50000) & 
                df['Category'].str.contains('game', case=False, na=False)]

        if df.empty:
            self.add_fallback(frame, "No apps meet the filter criteria for Task 6.")
            return

        df = df.sample(n=500) if len(df) > 500 else df

        fig = px.scatter(
            df, x='SizeMB', y='Rating', size='Installs', color='Category',
            title='App Size vs Rating (Bubble = Installs)',
            size_max=60, opacity=0.7,
            hover_data=['App'] if 'App' in df.columns else None
        )
        fig.show()
        self.add_fallback(frame, "Interactive bubble chart opened in browser.")

    def task7(self, frame, override=False):
        print("Running Task 7 (Heatmap)")
        current_hour = datetime.now(timezone('Asia/Kolkata')).hour
        if not self.view_all and not (14 <= current_hour < 16):
            self.add_fallback(frame, "This graph is only available between 2 PM and 4 PM IST.")
            return

        df = apps_df.copy()
        one_year_ago = pd.Timestamp.now() - pd.DateOffset(years=1)
        df = df[df['Installs'] >= 100000]
        df = df[df['Reviews'] > 1000]
        if 'Genres' in df.columns:
            df = df[~df['Genres'].str.startswith(tuple('AFEGIK'))]
        else:
            df = df[~df['Category'].str.startswith(tuple('AFEGIK'))]

        if df.empty:
            self.add_fallback(frame, "No data available for correlation heatmap.")
            return

        corr_matrix = df[['Installs', 'Rating', 'Reviews']].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
        ax.set_title("Correlation Matrix: Installs, Rating, Reviews")
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=1, fill='both')
        plt.close(fig)

if __name__ == "__main__":
    app = PlayStoreDashboard()
    app.mainloop()
