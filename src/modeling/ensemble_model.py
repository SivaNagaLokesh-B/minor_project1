import tensorflow as tf
from tensorflow.keras import layers, models
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

class PhysicsInformedTCN(tf.keras.Model):
    def __init__(self, input_shape, num_failure_modes=3):
        super(PhysicsInformedTCN, self).__init__()
        self.tcn_layers = self._build_tcn(input_shape)
        self.physics_constraint = PhysicsConstraintLayer(num_failure_modes)
        self.rul_head = layers.Dense(1, activation='relu', name='rul_prediction')
        self.failure_prob_head = layers.Dense(1, activation='sigmoid', name='failure_probability')
        
    def _build_tcn(self, input_shape):
        """Build Temporal Convolutional Network layers"""
        return tf.keras.Sequential([
            layers.Conv1D(64, 3, padding='causal', activation='relu', input_shape=input_shape),
            layers.Conv1D(128, 3, padding='causal', activation='relu'),
            layers.Conv1D(256, 3, padding='causal', activation='relu'),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.2)
        ])
    
    def call(self, inputs):
        x = self.tcn_layers(inputs)
        x = self.physics_constraint(x)
        rul = self.rul_head(x)
        failure_prob = self.failure_prob_head(x)
        return {'rul': rul, 'failure_probability': failure_prob}

class PhysicsConstraintLayer(layers.Layer):
    def __init__(self, num_failure_modes):
        super(PhysicsConstraintLayer, self).__init__()
        self.num_failure_modes = num_failure_modes
        
    def call(self, inputs):
        # Apply physics-based constraints
        # Example: Ensure RUL predictions are positive and follow degradation patterns
        return tf.nn.relu(inputs)  # Simple constraint for demonstration

class PredictiveMaintenanceEnsemble:
    def __init__(self, model_config):
        self.config = model_config
        self.models = {}
        self.weights = model_config['ensemble']['weights']
        
    def build_models(self, input_shape, num_features):
        """Build all ensemble models"""
        # Neural Network models
        self.models['tcn_physics'] = PhysicsInformedTCN(input_shape)
        self.models['lstm_attention'] = self._build_lstm_attention(input_shape)
        
        # Tree-based models
        self.models['gradient_boosting'] = xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1
        )
        self.models['random_forest'] = RandomForestRegressor(n_estimators=50)
        
    def _build_lstm_attention(self, input_shape):
        """Build LSTM with attention mechanism"""
        inputs = tf.keras.Input(shape=input_shape)
        x = layers.LSTM(128, return_sequences=True)(inputs)
        x = layers.LSTM(64, return_sequences=True)(x)
        
        # Attention mechanism
        attention = layers.Attention()([x, x])
        x = layers.GlobalAveragePooling1D()(attention)
        x = layers.Dense(32, activation='relu')(x)
        
        rul = layers.Dense(1, activation='relu', name='rul')(x)
        failure_prob = layers.Dense(1, activation='sigmoid', name='failure_probability')(x)
        
        return tf.keras.Model(inputs=inputs, outputs=[rul, failure_prob])
    
    def predict_ensemble(self, X):
        """Make ensemble prediction"""
        predictions = []
        
        for model_name, model in self.models.items():
            if hasattr(model, 'predict'):
                # Scikit-learn style model
                pred = model.predict(X)
            else:
                # TensorFlow model
                pred = model(X, training=False)['rul'].numpy()
                
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = np.average(predictions, weights=self.weights, axis=0)
        return ensemble_pred