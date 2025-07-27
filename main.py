import pandas as pd
import numpy as np
import argparse
import joblib
import logging
import warnings
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import scipy.stats as stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('distance_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class DistancePredictionSystem:
    
    def __init__(self, output_dir='results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.models = {}
        self.best_model = None
        self.preprocessor = None
        self.results = {}
        
    def load_and_validate_data(self, filepath):
        logger.info(f"Loading data from {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.strip()
            
            logger.info(f"Dataset shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            
            if 'distance' not in df.columns:
                raise ValueError("Target variable 'distance' not found in dataset")
            
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                logger.warning("Missing values detected:")
                for col, missing in missing_data[missing_data > 0].items():
                    logger.warning(f"  {col}: {missing} missing values")
            
            logger.info("Target variable statistics:")
            logger.info(f"  Mean distance: {df['distance'].mean():.2f}")
            logger.info(f"  Std distance: {df['distance'].std():.2f}")
            logger.info(f"  Min distance: {df['distance'].min():.2f}")
            logger.info(f"  Max distance: {df['distance'].max():.2f}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def exploratory_data_analysis(self, df):
        logger.info("Performing Exploratory Data Analysis...")
        
        fig_dir = self.output_dir / 'figures'
        fig_dir.mkdir(exist_ok=True)
        
        plt.style.use('seaborn-v0_8')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].hist(df['distance'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distance Distribution')
        axes[0, 0].set_xlabel('Distance')
        axes[0, 0].set_ylabel('Frequency')
        
        axes[0, 1].boxplot(df['distance'])
        axes[0, 1].set_title('Distance Box Plot')
        axes[0, 1].set_ylabel('Distance')
        
        stats.probplot(df['distance'], dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
        
        axes[1, 1].hist(np.log1p(df['distance']), bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1, 1].set_title('Log-transformed Distance Distribution')
        axes[1, 1].set_xlabel('Log(Distance + 1)')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'target_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.drop('distance')
        if len(numerical_cols) > 1:
            correlation_matrix = df[numerical_cols].corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                        square=True, fmt='.2f')
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(fig_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            n_cats = len(categorical_cols)
            rows = (n_cats + 1) // 2
            cols = 2 if n_cats > 1 else 1
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
            
            if n_cats > 1:
                axes = axes.flatten()
            else:
                axes = [axes]
            
            for i, col in enumerate(categorical_cols):
                if i < len(axes):
                    df.groupby(col)['distance'].mean().plot(kind='bar', ax=axes[i])
                    axes[i].set_title(f'Average Distance by {col}')
                    axes[i].set_ylabel('Average Distance')
                    axes[i].tick_params(axis='x', rotation=45)
            
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(fig_dir / 'categorical_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"EDA plots saved to {fig_dir}")
    
    def prepare_features(self, df):
        logger.info("Preparing features...")
        
        X = df.drop('distance', axis=1)
        y = df['distance']
        
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"Categorical features: {categorical_features}")
        logger.info(f"Numerical features: {numerical_features}")
        
        from sklearn.impute import SimpleImputer
        
        preprocessor_steps = []
        
        if categorical_features:
            preprocessor_steps.append(('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features))
        
        if numerical_features:
            preprocessor_steps.append(('num', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), numerical_features))
        
        self.preprocessor = ColumnTransformer(
            transformers=preprocessor_steps,
            remainder='drop'
        )
        
        return X, y
    
    def train_multiple_models(self, X, y):
        logger.info("Training multiple models...")
        
        models_config = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVM': SVR(kernel='rbf', C=1.0)
        }
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        results_summary = []
        
        for name, model in models_config.items():
            logger.info(f"Training {name}...")
            
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('model', model)
            ])
            
            cv_scores = cross_val_score(pipeline, X_train, y_train, 
                                      cv=5, scoring='r2', n_jobs=-1)
            
            pipeline.fit(X_train, y_train)
            
            y_pred_train = pipeline.predict(X_train)
            y_pred_test = pipeline.predict(X_test)
            
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            self.models[name] = {
                'pipeline': pipeline,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'y_pred_test': y_pred_test
            }
            
            results_summary.append({
                'Model': name,
                'CV R² (mean ± std)': f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}",
                'Test R²': f"{test_r2:.3f}",
                'Test RMSE': f"{test_rmse:.3f}",
                'Test MAE': f"{test_mae:.3f}"
            })
            
            logger.info(f"{name} - Test R²: {test_r2:.3f}, RMSE: {test_rmse:.3f}")
        
        self.results['model_comparison'] = pd.DataFrame(results_summary)
        self.results['test_data'] = (X_test, y_test)
        
        best_model_name = max(self.models.keys(), 
                            key=lambda x: self.models[x]['cv_r2_mean'])
        self.best_model = self.models[best_model_name]
        self.best_model['name'] = best_model_name
        
        logger.info(f"Best model: {best_model_name}")
        return self.results['model_comparison']
    
    def hyperparameter_tuning(self, X, y, model_name='Random Forest'):
        logger.info(f"Performing hyperparameter tuning for {model_name}...")
        
        if model_name == 'Random Forest':
            param_grid = {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [None, 10, 20],
                'model__min_samples_split': [2, 5, 10]
            }
            base_model = RandomForestRegressor(random_state=42)
        else:
            logger.warning(f"Hyperparameter tuning not available for {model_name}")
            return None
        
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('model', base_model)
        ])
        
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.3f}")
        
        if grid_search.best_score_ > self.best_model['cv_r2_mean']:
            self.best_model = {
                'pipeline': grid_search.best_estimator_,
                'cv_r2_mean': grid_search.best_score_,
                'name': f'{model_name} (Tuned)',
                'best_params': grid_search.best_params_
            }
            logger.info("Updated best model with tuned parameters")
        
        return grid_search
    
    def create_visualizations(self):
        logger.info("Creating model evaluation visualizations...")
        
        fig_dir = self.output_dir / 'figures'
        X_test, y_test = self.results['test_data']
        
        comparison_df = self.results['model_comparison'].copy()
        comparison_df['Test R²'] = comparison_df['Test R²'].astype(float)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(comparison_df['Model'], comparison_df['Test R²'], 
                      color='skyblue', alpha=0.7, edgecolor='black')
        plt.title('Model Performance Comparison (Test R²)')
        plt.ylabel('R² Score')
        plt.xticks(rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        y_pred_best = self.best_model['y_pred_test'] if 'y_pred_test' in self.best_model else \
                     self.best_model['pipeline'].predict(X_test)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test, y_pred_best, alpha=0.6, color='blue')
        
        min_val = min(y_test.min(), y_pred_best.min())
        max_val = max(y_test.max(), y_pred_best.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Distance')
        plt.ylabel('Predicted Distance')
        plt.title(f'Actual vs Predicted - {self.best_model["name"]}')
        plt.legend()
        
        r2 = r2_score(y_test, y_pred_best)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'actual_vs_predicted.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        residuals = y_test - y_pred_best
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        axes[0].scatter(y_pred_best, residuals, alpha=0.6, color='green')
        axes[0].axhline(y=0, color='red', linestyle='--')
        axes[0].set_xlabel('Predicted Distance')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted')
        
        axes[1].hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residual Distribution')
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'residual_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        if hasattr(self.best_model['pipeline'].named_steps['model'], 'feature_importances_'):
            self.plot_feature_importance()
        
        logger.info(f"Visualizations saved to {fig_dir}")
    
    def plot_feature_importance(self):
        model = self.best_model['pipeline'].named_steps['model']
        preprocessor = self.best_model['pipeline'].named_steps['preprocessor']
        
        feature_names = []
        
        if hasattr(preprocessor, 'named_transformers_'):
            if 'cat' in preprocessor.named_transformers_:
                cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
                feature_names.extend(cat_features)
            
            if 'num' in preprocessor.named_transformers_:
                num_features = preprocessor.transformers_[1][2]
                feature_names.extend(num_features)
        
        importances = model.feature_importances_
        
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'feature_importance.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_model_and_results(self):
        logger.info("Saving model and results...")
        
        model_path = self.output_dir / 'best_model.pkl'
        joblib.dump(self.best_model['pipeline'], model_path)
        logger.info(f"Best model saved to {model_path}")
        
        results_path = self.output_dir / 'model_comparison_results.csv'
        self.results['model_comparison'].to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")
        
        detailed_results = {
            'best_model_name': self.best_model['name'],
            'best_model_cv_score': self.best_model['cv_r2_mean'],
            'timestamp': datetime.now().isoformat()
        }
        
        if 'best_params' in self.best_model:
            detailed_results['best_params'] = self.best_model['best_params']
        
        import json
        with open(self.output_dir / 'detailed_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2)
    
    def predict_new_data(self, filepath, output_path=None):
        logger.info(f"Making predictions on {filepath}")
        
        try:
            new_data = pd.read_csv(filepath)
            new_data.columns = new_data.columns.str.strip()
            
            predictions = self.best_model['pipeline'].predict(new_data)
            
            new_data['Predicted_Distance'] = predictions
            
            if output_path is None:
                output_path = self.output_dir / 'predictions.csv'
            
            new_data.to_csv(output_path, index=False)
            logger.info(f"Predictions saved to {output_path}")
            
            return new_data
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Enhanced Distance Prediction System')
    parser.add_argument('--train', type=str, help='CSV file to train models on')
    parser.add_argument('--predict', type=str, help='CSV file to predict distances for')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output file for predictions')
    parser.add_argument('--tune', action='store_true', help='Perform hyperparameter tuning')
    parser.add_argument('--results-dir', type=str, default='results', help='Directory to save results')
    
    args = parser.parse_args()
    
    system = DistancePredictionSystem(output_dir=args.results_dir)
    
    if args.train:
        df = system.load_and_validate_data(args.train)
        
        system.exploratory_data_analysis(df)
        
        X, y = system.prepare_features(df)
        
        comparison_results = system.train_multiple_models(X, y)
        print("\n📊 Model Comparison Results:")
        print(comparison_results.to_string(index=False))
        
        if args.tune:
            system.hyperparameter_tuning(X, y)
        
        system.create_visualizations()
        
        system.save_model_and_results()
        
        print(f"\n✅ Training completed successfully!")
        print(f"📁 Results saved to: {system.output_dir}")
        print(f"🏆 Best model: {system.best_model['name']}")
        print(f"📈 Best CV R² score: {system.best_model['cv_r2_mean']:.3f}")
    
    if args.predict:
        if not system.best_model:
            model_path = Path(args.results_dir) / 'best_model.pkl'
            if model_path.exists():
                system.best_model = {'pipeline': joblib.load(model_path)}
                logger.info(f"Loaded existing model from {model_path}")
            else:
                raise ValueError("No trained model found. Please train a model first.")
        
        predictions_df = system.predict_new_data(args.predict, args.output)
        print(f"\n📈 Predictions completed and saved to: {args.output}")
        print(f"📊 Predicted {len(predictions_df)} distances")

if __name__ == "__main__":
    main()