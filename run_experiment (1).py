import os
import re
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.contingency_tables import mcnemar
from groq import Groq
from dotenv import load_dotenv

# ===== API SETUP =====
load_dotenv()

API_KEYS = [v for k, v in sorted(os.environ.items()) if k.startswith("GROQ_API_KEY") and v and "your_" not in v]
if not API_KEYS:
    raise ValueError("No valid GROQ API keys found in .env!")

current_key_index = 0
client = Groq(api_key=API_KEYS[current_key_index])
print(f"✅ Loaded {len(API_KEYS)} API key(s)")

def rotate_key():
    global current_key_index, client
    current_key_index = (current_key_index + 1) % len(API_KEYS)
    client = Groq(api_key=API_KEYS[current_key_index])
    print(f"  🔄 Rotated to API key #{current_key_index + 1}")

# ===== DATASET LOAD =====
df = pd.read_excel("GSM-Plus-BN.xlsx")
reverse_df = df[df['perturbation_type'] == 'reversing operation'].copy()
reverse_df = reverse_df.dropna(subset=['Bangla_Question', 'Bangla Seed_question', 'seed_answer'])
reverse_df = reverse_df.rename(columns={'Unnamed: 2': 'reverse_answer'})

print(f"✅ Total reversing operation samples: {len(reverse_df)}")

# ===== SAMPLE =====
sampled_df = reverse_df.sample(n=500, random_state=42).reset_index(drop=True)
print(f"✅ Sampled: {len(sampled_df)} rows")


# ===== MODELS =====
MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "meta-llama/llama-4-scout-17b-16e-instruct"
]

CONDITIONS = ["zero_shot", "few_shot", "cot"]

# ===== LOAD PREVIOUS RESULTS IF EXISTS =====
if os.path.exists("experiment_results.csv"):
    results_df_old = pd.read_csv("experiment_results.csv")
    results = results_df_old.to_dict('records')
    print(f"✅ Loaded {len(results)} previous results — continuing from where we left off!")
else:
    results = []
    print("🆕 Starting fresh experiment")

# ===== ALREADY DONE CHECK =====
def already_done(sample_id, model, condition):
    for r in results:
        if r['sample_id'] == sample_id and r['model'] == model and r['condition'] == condition:
            return True
    return False

# ===== PROMPT FUNCTIONS =====
def zero_shot_prompt(question):
    return f"""নিচের গণিতের সমস্যাটি সমাধান করো। শুধুমাত্র চূড়ান্ত উত্তরটি দাও #### চিহ্নের পরে।

প্রশ্ন: {question}

উত্তর:"""

FEW_SHOT_EXAMPLES = """
প্রশ্ন: রাহেলার কাছে ১০টি আপেল ছিল। সে ৩টি খেয়েছে। বাকি কতটি আছে?
উত্তর: #### ৭

প্রশ্ন: একটি দোকানে ৫০টি বই ছিল। ২০টি বিক্রি হয়েছে। বাকি কতটি আছে?
উত্তর: #### ৩০

প্রশ্ন: করিম প্রতিদিন ১২টি রুটি বানায়। সে ৪টি খায়। বাকি কতটি থাকে?
উত্তর: #### ৮
"""

def few_shot_prompt(question):
    return f"""নিচের উদাহরণগুলো দেখো এবং একইভাবে সমস্যাটি সমাধান করো। শুধুমাত্র চূড়ান্ত উত্তরটি দাও #### চিহ্নের পরে।

{FEW_SHOT_EXAMPLES}

প্রশ্ন: {question}
উত্তর:"""

def cot_prompt(question):
    return f"""নিচের গণিতের সমস্যাটি সমাধান করো। প্রথমে বাংলায় ধাপে ধাপে চিন্তা করো, তারপর #### চিহ্নের পরে চূড়ান্ত উত্তর দাও।

প্রশ্ন: {question}

ধাপে ধাপে সমাধান:"""

# ===== API CALL =====
def call_groq(prompt, model="llama-3.3-70b-versatile", temperature=0, retries=None):
    if retries is None:
        retries = len(API_KEYS) * 2
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=512
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e).lower()
            print(f"  ⚠️ API Error (attempt {attempt+1}): {e}")
            if "rate_limit" in err or "429" in err or "quota" in err:
                if len(API_KEYS) > 1:
                    rotate_key()
                else:
                    print(f"  ⏳ Waiting 65 seconds...")
                    time.sleep(65)
            elif attempt < retries - 1:
                time.sleep(10)
            else:
                print("  ❌ All retries failed, skipping this sample")
                return None
    return None

# ===== ANSWER EXTRACTION =====
def normalize_answer(answer_str):
    if answer_str is None:
        return None
    try:
        bangla_digits = '০১২৩৪৫৬৭৮৯'
        s = str(answer_str)
        for i, d in enumerate(bangla_digits):
            s = s.replace(d, str(i))
        s = s.replace(',', '').strip()
        return float(s)
    except:
        return None

def extract_answer(model_output):
    if model_output is None:
        return None
    match = re.search(r'####\s*([\d০-৯.,]+)', model_output)
    if match:
        return normalize_answer(match.group(1))
    numbers = re.findall(r'[\d০-৯.,]+', model_output)
    if numbers:
        return normalize_answer(numbers[-1])
    return None

# ===== MAIN EXPERIMENT LOOP =====
total = len(MODELS) * len(CONDITIONS) * len(sampled_df)
done = 0

print(f"\n🚀 Total samples needed: {total}")
print(f"⏭️  Skipping already completed samples...\n")

for model in MODELS:
    print(f"\n🤖 Model: {model}")

    for condition in CONDITIONS:
        print(f"  📝 Condition: {condition}")

        for idx, row in sampled_df.iterrows():

            # Skip if already done
            if already_done(idx, model, condition):
                done += 1
                continue

            forward_q = row['Bangla Seed_question']
            reverse_q = row['Bangla_Question']
            true_fwd = normalize_answer(str(row['seed_answer']))
            true_rev = normalize_answer(str(row['reverse_answer']))

            # Prompt select
            if condition == "zero_shot":
                fwd_prompt = zero_shot_prompt(forward_q)
                rev_prompt = zero_shot_prompt(reverse_q)
            elif condition == "few_shot":
                fwd_prompt = few_shot_prompt(forward_q)
                rev_prompt = few_shot_prompt(reverse_q)
            else:
                fwd_prompt = cot_prompt(forward_q)
                rev_prompt = cot_prompt(reverse_q)

            # Forward
            fwd_output = call_groq(fwd_prompt, model=model)
            fwd_answer = extract_answer(fwd_output)
            fwd_correct = (fwd_answer == true_fwd) if (fwd_answer is not None and true_fwd is not None) else False
            time.sleep(2)

            # Reverse
            rev_output = call_groq(rev_prompt, model=model)
            rev_answer = extract_answer(rev_output)
            rev_correct = (rev_answer == true_rev) if (rev_answer is not None and true_rev is not None) else False
            time.sleep(2)

            results.append({
                "sample_id": idx,
                "model": model,
                "condition": condition,
                "forward_correct": fwd_correct,
                "reverse_correct": rev_correct,
                "forward_answer": fwd_answer,
                "reverse_answer": rev_answer,
                "true_fwd": true_fwd,
                "true_rev": true_rev,
                "forward_output": fwd_output,
                "reverse_output": rev_output
            })

            done += 1
            if done % 25 == 0:
                print(f"    ⏳ Progress: {done}/{total} done")
                pd.DataFrame(results).to_csv("experiment_results.csv", index=False)

print("\n💾 Saving final results...")
results_df = pd.DataFrame(results)
results_df.to_csv("experiment_results.csv", index=False)
print("✅ experiment_results.csv saved!")

# ===== ANALYSIS =====
summary = results_df.groupby(['model', 'condition']).agg(
    forward_acc=('forward_correct', 'mean'),
    reverse_acc=('reverse_correct', 'mean'),
    n=('sample_id', 'count')
).reset_index()

summary['RAG_score'] = summary['forward_acc'] - summary['reverse_acc']
summary.to_csv("summary_results.csv", index=False)

print("\n📊 Results Summary:")
print(summary.to_string())

# McNemar's Test
print("\n📐 McNemar's Test:")
for model in MODELS:
    for condition in CONDITIONS:
        subset = results_df[(results_df['model'] == model) &
                            (results_df['condition'] == condition)]
        fwd = subset['forward_correct'].astype(int).values
        rev = subset['reverse_correct'].astype(int).values
        b = int(sum((fwd == 1) & (rev == 0)))
        c = int(sum((fwd == 0) & (rev == 1)))
        if b + c > 0:
            table = [[0, b], [c, 0]]
            result = mcnemar(table, exact=True)
            sig = '✅ Significant' if result.pvalue < 0.05 else '❌ Not significant'
            print(f"  {model} | {condition} → p={result.pvalue:.4f} {sig}")

# ===== VISUALIZATION =====
pivot = summary.pivot(index='model', columns='condition', values='RAG_score')
plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r', center=0, linewidths=0.5)
plt.title('Reasoning Asymmetry Gap (RAG Score)\nHigher = More Asymmetric', fontsize=13)
plt.tight_layout()
plt.savefig("rag_heatmap.png", dpi=300)
plt.close()
print("\n✅ rag_heatmap.png saved!")

best_model = "llama-3.3-70b-versatile"
best_data = summary[summary['model'] == best_model].sort_values('condition')

x = range(len(CONDITIONS))
width = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar([i - width/2 for i in x], best_data['forward_acc'], width, label='Forward Accuracy', color='steelblue')
ax.bar([i + width/2 for i in x], best_data['reverse_acc'], width, label='Reverse Accuracy', color='tomato')
ax.set_xticks(x)
ax.set_xticklabels(CONDITIONS)
ax.set_ylabel('Accuracy')
ax.set_title(f'Forward vs Reverse Accuracy — {best_model}')
ax.legend()
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("forward_vs_reverse.png", dpi=300)
plt.close()
print("✅ forward_vs_reverse.png saved!")

# ===== FAILURE CASES =====
failures = results_df[
    (results_df['model'] == best_model) &
    (results_df['condition'] == 'cot') &
    (results_df['reverse_correct'] == False)
].head(100)

failures[['sample_id', 'reverse_output', 'true_rev', 'reverse_answer']].to_csv("failure_cases.csv", index=False)
print(f"✅ failure_cases.csv saved! ({len(failures)} cases)")

print("\n🎉 Experiment complete! Files saved:")
print("  - experiment_results.csv")
print("  - summary_results.csv")
print("  - rag_heatmap.png")
print("  - forward_vs_reverse.png")
print("  - failure_cases.csv")

