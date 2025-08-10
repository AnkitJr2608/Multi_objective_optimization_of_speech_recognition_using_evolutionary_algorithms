import random
import numpy as np
import librosa
import matplotlib.pyplot as plt
import torch
from deap import algorithms, base, creator, tools
from sklearn.metrics import accuracy_score
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# -------------------------------
# Toggle between fake and real ASR
# -------------------------------
USE_FAKE_ASR = False  # Set to False to run real wav2vec2 ASR (slower)

# -------------------------------
# Load pretrained wav2vec2 model and processor (only if real ASR)
# -------------------------------
if not USE_FAKE_ASR:
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    model.eval()

# -------------------------------
# Audio Data
# -------------------------------
audio_paths = [
    r"C:\Users\ANKIT SINGH\Documents\recording_1_20250412_193115.wav",
    r"C:\Users\ANKIT SINGH\Documents\recording_2_20250412_193119.wav",
    r"C:\Users\ANKIT SINGH\Documents\recording_3_20250412_193124.wav"
]
transcripts = ["hello world", "hello noisy", "good morning"]  # Ground truth

# -------------------------------
# 1. Multi-objective fitness: (min WER, min Latency, max Robustness)
# -------------------------------
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# -------------------------------
# 2. Toolbox setup
# -------------------------------
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=10)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# -------------------------------
# 3. Audio preprocessing helper
# -------------------------------
def preprocess_audio(y, noise_reduction_strength, audio_gain):
    y = y * audio_gain
    y = y * noise_reduction_strength
    return y

# -------------------------------
# 4a. Fake ASR model for quick demo
# -------------------------------
def fake_asr_model(mfcc_features, individual):
    options = ["hello world", "good morning", "hello noisy"]
    return random.choice(options)

# -------------------------------
# 4b. Real wav2vec2 ASR with individual hyperparams
# -------------------------------
def wav2vec2_asr_with_params(audio_path, individual):
    clipped_ind = [min(max(gene, 0.0), 1.0) for gene in individual]

    logit_threshold = clipped_ind[0]
    noise_reduction_strength = clipped_ind[4] 
    audio_gain = 0.5 + clipped_ind[5] * 1.5

    y, sr = librosa.load(audio_path, sr=16000)
    y = preprocess_audio(y, noise_reduction_strength, audio_gain)
    input_values = processor(y, sampling_rate=16000, return_tensors="pt").input_values

    with torch.no_grad():
        logits = model(input_values).logits

    threshold_val = torch.quantile(logits, logit_threshold)
    logits_masked = torch.where(logits >= threshold_val, logits, torch.tensor(float('-inf')))
    predicted_ids = torch.argmax(logits_masked, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    return transcription.lower()

# -------------------------------
# 5. Simulated latency and robustness functions
# -------------------------------
def simulate_latency(individual):
    return sum(individual) * 100  # ms

def test_noisy_audio(individual):
    return random.uniform(0.5, 1.0)

# -------------------------------
# 6. Evaluation function
# -------------------------------
LATENCY_THRESHOLD = 800  # ms

def evaluate(individual):
    if USE_FAKE_ASR:
        predicted_texts = [fake_asr_model(None, individual) for _ in audio_paths]
    else:
        predicted_texts = [wav2vec2_asr_with_params(path, individual) for path in audio_paths]

    wer = 1 - accuracy_score(transcripts, predicted_texts)
    latency = simulate_latency(individual)
    robustness = test_noisy_audio(individual)

    if latency > LATENCY_THRESHOLD:
        wer += 0.1

    return wer, latency, robustness

toolbox.register("evaluate", evaluate)

# -------------------------------
# 7. Custom mutation: Gaussian + clipping
# -------------------------------
def mutate_and_clip(individual):
    tools.mutGaussian(individual, mu=0, sigma=0.2, indpb=0.1)
    for i in range(len(individual)):
        individual[i] = max(0.0, min(1.0, individual[i]))
    return individual,

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", mutate_and_clip)
toolbox.register("select", tools.selNSGA2)

# -------------------------------
# 8. Main execution and visualization
# -------------------------------
def main():
    pop = toolbox.population(n=50)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)

    algorithms.eaMuPlusLambda(pop, toolbox, mu=50, lambda_=100,
                              cxpb=0.7, mutpb=0.3, ngen=50,
                              stats=stats, halloffame=hof, verbose=True)

    print("\nPareto-optimal solutions (WER vs Latency vs Robustness):")
    for ind in hof:
        print(f"Params: {ind}\nFitness: {ind.fitness.values}\n")

    pareto_wer = [ind.fitness.values[0] for ind in hof]
    pareto_latency = [ind.fitness.values[1] for ind in hof]
    pareto_robustness = [ind.fitness.values[2] for ind in hof]

    plt.figure(figsize=(8, 6))
    plt.scatter(pareto_latency, pareto_wer, c=pareto_robustness, cmap="viridis", s=80, edgecolor='k')
    plt.colorbar(label="Robustness Score")
    plt.xlabel("Latency (ms)")
    plt.ylabel("WER")
    plt.title("Pareto Front: WER vs Latency (colored by Robustness)")
    plt.grid(True)
    plt.show()

    return pop, hof

if __name__ == "__main__":
    pop, hof = main()
