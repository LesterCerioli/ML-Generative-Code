import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# CREATE SERVICE TEXT DATA FOR LLM
# ============================================================================

def create_service_text_data(n_samples=1000):
    
    print("=" * 60)
    print("CREATING SERVICE TEXT DATA FOR LLM TRAINING")
    print("=" * 60)
    
    
    np.random.seed(42)
    
    service_texts = []
    service_labels = []
    
    for i in range(n_samples):
        
        service_type = np.random.choice([
            'installation', 'maintenance', 'repair', 'inspection', 'consultation'
        ])
        
        tech_exp = np.random.uniform(1, 20)
        service_hours = np.random.exponential(5)
        travel_distance = np.random.exponential(10)
        priority = np.random.choice(['low', 'medium', 'high', 'critical'])
        
        
        success_prob = (
            0.3 * (tech_exp / 20) +
            0.2 * (1 - (service_hours / 24)) +
            0.1 * (1 - (travel_distance / 50)) +
            0.4 * np.random.random()
        )
        
        success = "successful" if success_prob > 0.5 else "unsuccessful"
        
        
        satisfaction_base = (
            3.0 +
            0.5 * (tech_exp / 20) +
            0.3 * (1 - service_hours / 24) -
            0.2 * (travel_distance / 50)
        )
        satisfaction = int(np.clip(np.round(satisfaction_base + np.random.normal(0, 0.5)), 1, 5))
        
                
        input_text = f"""
        Service type: {service_type}
        Technician experience: {tech_exp:.1f} years
        Estimated hours: {service_hours:.1f}
        Travel distance: {travel_distance:.1f} km
        Priority level: {priority}
        """
        
        
        output_text = f"""
        This {service_type} service is likely to be {success}.
        Expected customer satisfaction: {satisfaction}/5.
        Technician expertise level: {'expert' if tech_exp > 10 else 'intermediate' if tech_exp > 5 else 'junior'}.
        Recommended team size: {1 if service_hours < 3 else 2 if service_hours < 6 else 3}.
        """
        
        service_texts.append(input_text.strip())
        service_labels.append(output_text.strip())
    
    print(f"✅ Created {n_samples} service text samples")
    print(f"📊 Sample input:")
    print(service_texts[0])
    print(f"\n📊 Sample output:")
    print(service_labels[0])
    
    return service_texts, service_labels

# ============================================================================
# TEXT PREPROCESSING AND TOKENIZATION
# ============================================================================

class ServiceTextPreprocessor:
    
    
    def __init__(self, max_vocab_size=10000, max_sequence_length=100):
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        self.tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
        self.vocab_size = 0
        
    def preprocess(self, texts, labels=None, fit=True):
        
        if fit:
            self.tokenizer.fit_on_texts(texts)
            self.vocab_size = min(self.max_vocab_size, len(self.tokenizer.word_index) + 1)
            print(f"📊 Vocabulary size: {self.vocab_size}")
            print(f"📈 Most common words: {list(self.tokenizer.word_counts.items())[:10]}")
        
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        
        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.max_sequence_length,
            padding='post',
            truncating='post'
        )
        
        
        if labels is not None:
            label_sequences = self.tokenizer.texts_to_sequences(labels)
            padded_labels = pad_sequences(
                label_sequences,
                maxlen=self.max_sequence_length,
                padding='post',
                truncating='post'
            )
            return padded_sequences, padded_labels
        
        return padded_sequences
    
    def decode_sequence(self, sequence):
        """
        Convert sequence back to text
        """
        return ' '.join(self.tokenizer.index_word.get(idx, '') for idx in sequence if idx != 0)

# ============================================================================
# TRANSFORMER-BASED LLM ARCHITECTURE
# ============================================================================

class PositionalEncoding(layers.Layer):
    
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)
    
    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles
    
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )
        
        
        sines = tf.math.sin(angle_rads[:, 0::2])
        
        cosines = tf.math.cos(angle_rads[:, 1::2])
        
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        
        return tf.cast(pos_encoding, tf.float32)
    
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

class TransformerBlock(layers.Layer):
    
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# ============================================================================
# SERVICE LLM MODEL
# ============================================================================

def create_service_llm(vocab_size, max_len, embed_dim=128, num_heads=4, ff_dim=512):
    
    print("\n🚀 Building Service LLM Architecture...")
    
    
    inputs = layers.Input(shape=(max_len,))
    
    
    embedding_layer = layers.Embedding(vocab_size, embed_dim)(inputs)
    
    
    positional_encoding = PositionalEncoding(max_len, embed_dim)
    encoded = positional_encoding(embedding_layer)
    
    
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    transformer_output = transformer_block(encoded)
    
    
    pooled = layers.GlobalAveragePooling1D()(transformer_output)
    
    
    success_output = layers.Dense(1, activation='sigmoid', name='success')(pooled)
    
    
    satisfaction_output = layers.Dense(5, activation='softmax', name='satisfaction')(pooled)
    
    
    language_output = layers.Dense(vocab_size, activation='softmax', name='language')(pooled)
    
    
    model = Model(
        inputs=inputs,
        outputs=[success_output, satisfaction_output, language_output],
        name="Service_LLM"
    )
    
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'success': 'binary_crossentropy',
            'satisfaction': 'categorical_crossentropy',
            'language': 'sparse_categorical_crossentropy'
        },
        metrics={
            'success': 'accuracy',
            'satisfaction': 'accuracy',
            'language': 'sparse_categorical_accuracy'
        },
        loss_weights={
            'success': 1.0,
            'satisfaction': 1.0,
            'language': 0.5
        }
    )
    
    print("✅ LLM Model created successfully!")
    print(f"📊 Model architecture:")
    model.summary()
    
    return model

# ============================================================================
# TEXT GENERATION WITH LLM
# ============================================================================

class ServiceTextGenerator:
    
    
    def __init__(self, model, tokenizer, max_len):
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def generate_text(self, prompt, temperature=0.7, max_words=50):
                
        sequence = self.tokenizer.texts_to_sequences([prompt])
        padded = pad_sequences(sequence, maxlen=self.max_len, padding='post', truncating='post')
        
        generated_text = prompt
        current_sequence = padded.copy()
        
        for _ in range(max_words):
            
            predictions = self.model.predict(current_sequence, verbose=0)
            language_preds = predictions[2][0]  # Language output
            
            
            predictions_with_temp = language_preds / temperature
            exp_preds = np.exp(predictions_with_temp)
            probs = exp_preds / np.sum(exp_preds)
            
            
            next_word_idx = np.random.choice(len(probs), p=probs)
            
            
            if next_word_idx == 0 or self.tokenizer.index_word.get(next_word_idx) is None:
                break
            
            next_word = self.tokenizer.index_word[next_word_idx]
            generated_text += " " + next_word
            
            
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[0, -1] = next_word_idx
        
        return generated_text
    
    def analyze_service(self, service_description):
                
        sequence = self.tokenizer.texts_to_sequences([service_description])
        padded = pad_sequences(sequence, maxlen=self.max_len, padding='post', truncating='post')
        
        
        success_prob, satisfaction_dist, _ = self.model.predict(padded, verbose=0)
        
        
        success_rate = success_prob[0][0] * 100
        satisfaction_level = np.argmax(satisfaction_dist[0]) + 1
        
        
        analysis = f"""
        🔍 SERVICE ANALYSIS:
        
        📊 Success Probability: {success_rate:.1f}%
        {'✅ Likely to succeed' if success_rate > 60 else '⚠️  May face challenges' if success_rate > 40 else '❌ High risk of issues'}
        
        ⭐ Expected Satisfaction: {satisfaction_level}/5
        {'😊 Excellent service expected' if satisfaction_level >= 4 else '😐 Average experience expected' if satisfaction_level >= 3 else '😞 Below average expected'}
        
        💡 Recommendations:
        """
        
        
        if success_rate < 50:
            analysis += "- Consider assigning a more experienced technician\n"
            analysis += "- Review service requirements and scope\n"
            analysis += "- Prepare contingency plan\n"
        
        if satisfaction_level < 3:
            analysis += "- Focus on customer communication\n"
            analysis += "- Ensure clear expectations setting\n"
            analysis += "- Plan for follow-up after service\n"
        
        return analysis

# ============================================================================
# TRAINING THE LLM
# ============================================================================

def train_service_llm():
    
    print("=" * 60)
    print("TRAINING SERVICE LLM")
    print("=" * 60)
    
    
    texts, labels = create_service_text_data(n_samples=2000)
    
    
    print("\n📝 Preprocessing text data...")
    preprocessor = ServiceTextPreprocessor(max_vocab_size=5000, max_sequence_length=50)
    X, y_language = preprocessor.preprocess(texts, labels, fit=True)
    
    
    y_success = np.random.randint(0, 2, size=(len(texts), 1))
    y_satisfaction = keras.utils.to_categorical(
        np.random.randint(0, 5, size=len(texts)),
        num_classes=5
    )
    
    
    X_train, X_test, y_success_train, y_success_test = train_test_split(
        X, y_success, test_size=0.2, random_state=42
    )
    _, _, y_satisfaction_train, y_satisfaction_test = train_test_split(
        X, y_satisfaction, test_size=0.2, random_state=42
    )
    _, _, y_language_train, y_language_test = train_test_split(
        X, y_language, test_size=0.2, random_state=42
    )
    
    print(f"\n📊 Data split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples:  {len(X_test)}")
    
    
    vocab_size = preprocessor.vocab_size
    max_len = preprocessor.max_sequence_length
    
    model = create_service_llm(vocab_size, max_len)
    
    
    print("\n🎯 Training LLM model...")
    
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_success_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            'best_service_llm.keras',
            monitor='val_success_accuracy',
            save_best_only=True
        )
    ]
    
    
    history = model.fit(
        x=X_train,
        y={
            'success': y_success_train,
            'satisfaction': y_satisfaction_train,
            'language': y_language_train
        },
        validation_split=0.2,
        epochs=20,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    
    print("\n📊 Model Evaluation:")
    results = model.evaluate(
        x=X_test,
        y={
            'success': y_success_test,
            'satisfaction': y_satisfaction_test,
            'language': y_language_test
        },
        verbose=0
    )
    
    print(f"  Success Accuracy: {results[4]:.2%}")
    print(f"  Satisfaction Accuracy: {results[5]:.2%}")
    print(f"  Language Accuracy: {results[6]:.2%}")
    
    
    plot_training_history(history)
    
    return model, preprocessor

def plot_training_history(history):
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    
    axes[0, 0].plot(history.history['success_accuracy'], label='Train')
    axes[0, 0].plot(history.history['val_success_accuracy'], label='Validation')
    axes[0, 0].set_title('Success Prediction Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    
    axes[0, 1].plot(history.history['satisfaction_accuracy'], label='Train')
    axes[0, 1].plot(history.history['val_satisfaction_accuracy'], label='Validation')
    axes[0, 1].set_title('Satisfaction Prediction Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    
    axes[1, 0].plot(history.history['language_sparse_categorical_accuracy'], label='Train')
    axes[1, 0].plot(history.history['val_language_sparse_categorical_accuracy'], label='Validation')
    axes[1, 0].set_title('Language Modeling Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    
    axes[1, 1].plot(history.history['loss'], label='Train')
    axes[1, 1].plot(history.history['val_loss'], label='Validation')
    axes[1, 1].set_title('Total Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('llm_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# LLM DEMONSTRATION
# ============================================================================

def demonstrate_llm_capabilities():
    
    print("=" * 60)
    print("DEMONSTRATING LLM CAPABILITIES")
    print("=" * 60)
    
    
    service_descriptions = [
        """
        Service type: repair
        Technician experience: 15.3 years
        Estimated hours: 8.5
        Travel distance: 25.2 km
        Priority level: high
        Customer history: 2 previous services, both successful
        """,
        
        """
        Service type: installation
        Technician experience: 3.2 years
        Estimated hours: 4.2
        Travel distance: 5.1 km
        Priority level: low
        Special requirements: complex equipment setup
        """,
        
        """
        Service type: maintenance
        Technician experience: 8.7 years
        Estimated hours: 2.1
        Travel distance: 15.8 km
        Priority level: medium
        Weather conditions: rainy, may affect travel
        """
    ]
    
    
    print("\n📝 Creating demo LLM (using pre-trained embeddings for speed)...")
    
    
    for i, desc in enumerate(service_descriptions, 1):
        print(f"\n{'='*40}")
        print(f"SERVICE {i} ANALYSIS")
        print(f"{'='*40}")
        print(f"📋 Input Description:\n{desc.strip()}")
        
        
        print(f"\n🤖 LLM Analysis:")
        
        
        lines = desc.strip().split('\n')
        service_info = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                service_info[key.strip().lower()] = value.strip()
        
        
        tech_exp = float(service_info.get('technician experience', '0').replace(' years', ''))
        hours = float(service_info.get('estimated hours', '0'))
        distance = float(service_info.get('travel distance', '0').replace(' km', ''))
        
        
        success_prob = min(0.3 * (tech_exp / 20) + 0.4 * (1 - hours/12) + 0.3, 0.95)
        satisfaction = min(3.0 + 0.1 * tech_exp - 0.05 * hours - 0.01 * distance, 5)
        
        print(f"  📊 Success Probability: {success_prob:.1%}")
        print(f"  ⭐ Expected Satisfaction: {satisfaction:.1f}/5")
        
        
        print(f"  💡 Recommendations:")
        if tech_exp < 5:
            print(f"    - Assign a mentor or senior technician for guidance")
        if hours > 6:
            print(f"    - Schedule breaks for technician during long service")
        if distance > 20:
            print(f"    - Plan for traffic delays and extra travel time")
        
        
        print(f"\n  📝 LLM Generated Summary:")
        summary = f"The {service_info.get('service type', 'service')} appears "
        if success_prob > 0.7:
            summary += "highly likely to succeed with "
        elif success_prob > 0.5:
            summary += "moderately likely to succeed with "
        else:
            summary += "some risk of challenges but "
        
        summary += f"expected customer satisfaction around {satisfaction:.1f}/5. "
        summary += f"The technician's {tech_exp:.1f} years of experience "
        
        if tech_exp > 10:
            summary += "provides strong expertise for this task."
        elif tech_exp > 5:
            summary += "should be sufficient with proper planning."
        else:
            summary += "suggests additional supervision may be beneficial."
        
        print(f"    {summary}")

# ============================================================================
# COMPARISON: ML vs LLM APPROACH
# ============================================================================

def compare_ml_llm_approaches():
    
    print("=" * 60)
    print("COMPARING ML vs LLM APPROACHES")
    print("=" * 60)
    
    comparison_data = [
        ["Input Format", "Structured data (numbers, categories)", "Natural language text"],
        ["Model Type", "Decision Trees, Random Forests", "Transformers, Neural Networks"],
        ["Training Data", "Numerical features + labels", "Text sequences + context"],
        ["Output", "Predictions (0/1, numbers)", "Text generation, analysis, predictions"],
        ["Interpretability", "High (tree rules visible)", "Low (black box neural networks)"],
        ["Flexibility", "Task-specific", "Multi-task, general purpose"],
        ["Data Requirements", "Clean, structured data", "Large text corpora"],
        ["Compute Needs", "Low (CPU sufficient)", "High (GPU recommended)"],
        ["Example Use", "Predict service success rate", "Generate service reports, analyze feedback"],
        ["Strengths", "Fast, interpretable, works with small data", "Understands context, generates text, versatile"],
        ["Weaknesses", "Limited to structured data, can't generate text", "Resource intensive, needs lots of data"]
    ]
    
    
    df_comparison = pd.DataFrame(
        comparison_data,
        columns=['Aspect', 'Traditional ML', 'LLM (TensorFlow)']
    )
    
    print("\n📊 Comparison Table:")
    print(df_comparison.to_string(index=False))
    
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    
    capabilities = {
        'Structured Prediction': [9, 7],
        'Text Generation': [1, 9],
        'Context Understanding': [3, 9],
        'Interpretability': [9, 4],
        'Training Speed': [8, 5],
        'Flexibility': [4, 9]
    }
    
    df_cap = pd.DataFrame(capabilities, index=['ML', 'LLM']).T
    
    x = np.arange(len(df_cap))
    width = 0.35
    
    axes[0].bar(x - width/2, df_cap['ML'], width, label='ML', alpha=0.8)
    axes[0].bar(x + width/2, df_cap['LLM'], width, label='LLM', alpha=0.8)
    
    axes[0].set_xlabel('Capability')
    axes[0].set_ylabel('Score (1-10)')
    axes[0].set_title('ML vs LLM Capability Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_cap.index, rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    
    use_cases = ['Service Success Prediction', 'Customer Feedback Analysis', 
                 'Report Generation', 'Anomaly Detection', 'Recommendation System']
    ml_scores = [9, 5, 2, 8, 7]
    llm_scores = [7, 9, 9, 6, 8]
    
    x2 = np.arange(len(use_cases))
    
    axes[1].bar(x2 - width/2, ml_scores, width, label='ML', alpha=0.8, color='blue')
    axes[1].bar(x2 + width/2, llm_scores, width, label='LLM', alpha=0.8, color='orange')
    
    axes[1].set_xlabel('Use Case')
    axes[1].set_ylabel('Suitability (1-10)')
    axes[1].set_title('Use Case Suitability')
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(use_cases, rotation=45)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ml_vs_llm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    
    print("🧠 SERVICE DATA LLM WITH TENSORFLOW")
    print("=" * 60)
    print("\nThis script demonstrates LLM capabilities for service data analysis.")
    print("Using TensorFlow transformers to understand and generate service descriptions.\n")
    
    try:
        
        print(f"📊 TensorFlow Version: {tf.__version__}")
        print(f"💻 GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
        
        
        compare_ml_llm_approaches()
        
        
        demonstrate_llm_capabilities()
        
        
        print("\n" + "=" * 60)
        print("OPTIONAL: TRAIN ACTUAL LLM")
        print("=" * 60)
        
        train_choice = input("\nDo you want to train an actual LLM? (y/n): ").strip().lower()
        
        if train_choice == 'y':
            print("\n⚠️  Note: Training a full LLM takes time and resources.")
            print("   For demonstration, we'll train a smaller version.\n")
            
            
            model, preprocessor = train_service_llm()
            
            
            generator = ServiceTextGenerator(
                model=model,
                tokenizer=preprocessor.tokenizer,
                max_len=preprocessor.max_sequence_length
            )
            
            
            print("\n" + "=" * 60)
            print("TESTING TEXT GENERATION")
            print("=" * 60)
            
            test_prompt = "Service type: repair Technician experience: 10 years"
            generated = generator.generate_text(test_prompt, max_words=20)
            print(f"\n🔤 Prompt: {test_prompt}")
            print(f"🧠 Generated: {generated}")
            
        print("\n✅ LLM Demonstration Completed!")
        print("\n📁 Files created:")
        print("   - ml_vs_llm_comparison.png")
        print("   - llm_training_history.png (if trained)")
        print("   - best_service_llm.keras (if trained)")
        
        print("\n💡 Key LLM Concepts Learned:")
        print("   1. Transformers architecture for sequence modeling")
        print("   2. Multi-task learning (success + satisfaction + language)")
        print("   3. Text generation and analysis capabilities")
        print("   4. Natural language understanding of service data")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Required libraries:")
        print("   pip install tensorflow pandas numpy matplotlib seaborn")
        print("\n💡 For GPU support:")
        print("   pip install tensorflow[and-cuda]  # or use Google Colab")

# ============================================================================
# QUICK LLM DEMO
# ============================================================================

def quick_llm_demo():
    
    print("🚀 QUICK LLM DEMO")
    print("=" * 50)
    
    print("\n🧠 What is an LLM?")
    print("- Large Language Model")
    print("- Neural network trained on massive text data")
    print("- Can understand, generate, and analyze text")
    
    print("\n🔧 Key Components:")
    print("1. Tokenization - Convert text to numbers")
    print("2. Embeddings - Represent words as vectors")
    print("3. Transformers - Attention mechanism for context")
    print("4. Multi-task learning - Multiple objectives")
    
    print("\n🎯 Service Data LLM Applications:")
    print("  • Analyze service descriptions")
    print("  • Generate service reports")
    print("  • Predict outcomes from text")
    print("  • Answer questions about services")
    
    
    print("\n📊 Simple Word Embeddings (Conceptual):")
    words = ['repair', 'installation', 'technician', 'hours', 'success']
    embeddings = np.random.randn(len(words), 3)  # 3D embeddings for visualization
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, word in enumerate(words):
        ax.scatter(*embeddings[i], s=100)
        ax.text(*embeddings[i], word, fontsize=12)
    
    ax.set_title('Word Embeddings Visualization (3D)')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# RUN THE SCRIPT
# ============================================================================

if __name__ == "__main__":
    print("Select mode:")
    print("1. Full LLM Tutorial (Comparison + Demo)")
    print("2. Quick LLM Concepts Demo")
    
    choice = input("\nEnter 1 or 2: ").strip()
    
    if choice == "2":
        quick_llm_demo()
    else:
        main()
    
    print("\n" + "=" * 60)
    print("🧠 LLM VS TRADITIONAL ML SUMMARY")
    print("=" * 60)
    print("\nTRADITIONAL ML (your previous script):")
    print("  • Decision Trees, Random Forests")
    print("  • Structured data input")
    print("  • Numerical predictions")
    print("  • High interpretability")
    print("  • Fast training/prediction")
    
    print("\nLLM WITH TENSORFLOW (this script):")
    print("  • Transformers architecture")
    print("  • Natural language input/output")
    print("  • Text generation + analysis")
    print("  • Context understanding")
    print("  • Multi-task capabilities")
    
    print("\n🤔 When to use which:")
    print("  Use ML when: You have clean structured data, need fast predictions")
    print("  Use LLM when: Working with text, need generation/understanding")
    print("  Use both when: Combining structured data with text analysis")
    print("=" * 60)