import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

def process_employee_highlights(company_df, employee_highlights_col='employee_highlights'):
    """Extract features from employee highlights text"""
    # Skip if the column doesn't exist
    if employee_highlights_col not in company_df.columns:
        print(f"Warning: {employee_highlights_col} column not found")
        return company_df
    
    # Initialize new feature columns
    company_df['elite_education_count'] = 0
    company_df['tech_experience_count'] = 0
    company_df['founder_experience_count'] = 0
    company_df['team_quality_score'] = 0
    
    # Define elite institutions and companies
    elite_education = ['Stanford', 'Harvard', 'MIT', 'Berkeley', 'Princeton', 'Yale', 
                      'University of Pennsylvania', 'Cornell', 'Columbia']
    tech_companies = ['Google', 'Apple', 'Microsoft', 'Amazon', 'Meta', 'Facebook', 
                      'Netflix', 'Uber', 'Airbnb', 'Spotify']
    
    # Process each company's employee highlights
    for idx, row in company_df.iterrows():
        # Skip if missing
        highlight_value = row.get(employee_highlights_col)
        if highlight_value is None or (isinstance(highlight_value, float) and pd.isna(highlight_value)):
            continue
            
            
        highlights = str(row[employee_highlights_col])
        
        # Count elite education mentions
        elite_edu_count = sum(1 for edu in elite_education if edu.lower() in highlights.lower())
        company_df.at[idx, 'elite_education_count'] = elite_edu_count
        company_df.at[idx, 'team_quality_score'] += elite_edu_count * 1.0
        
        # Count tech company experience
        tech_exp_count = sum(1 for company in tech_companies if company.lower() in highlights.lower())
        company_df.at[idx, 'tech_experience_count'] = tech_exp_count
        company_df.at[idx, 'team_quality_score'] += tech_exp_count * 1.2
        
        # Count founder experience
        founder_keywords = ['founder', 'co-founder', 'cofounder', 'ceo', 'chief executive']
        founder_count = sum(1 for keyword in founder_keywords if keyword.lower() in highlights.lower())
        company_df.at[idx, 'founder_experience_count'] = founder_count
        company_df.at[idx, 'team_quality_score'] += founder_count * 1.8
    
    print(f"Processed employee highlights for {company_df['team_quality_score'].sum()} companies")
    return company_df

def evaluate_model(model, X, y, cv=5):
    """Evaluate model using k-fold cross validation"""
    # Setup cross-validation
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Calculate cross-validation scores
    cv_auc_scores = cross_val_score(model, X, y, cv=kf, scoring='roc_auc')
    cv_precision_scores = cross_val_score(model, X, y, cv=kf, scoring='precision')
    cv_recall_scores = cross_val_score(model, X, y, cv=kf, scoring='recall')
    
    # Print cross-validation results
    print(f"Cross-Validation Results (k={cv}):")
    print(f"  AUC: {cv_auc_scores.mean():.4f} (±{cv_auc_scores.std():.4f})")
    print(f"  Precision: {cv_precision_scores.mean():.4f} (±{cv_precision_scores.std():.4f})")
    print(f"  Recall: {cv_recall_scores.mean():.4f} (±{cv_recall_scores.std():.4f})")
    
    return {
        'auc': cv_auc_scores.mean(),
        'precision': cv_precision_scores.mean(),
        'recall': cv_recall_scores.mean()
    }

def score_companies(company_data_path, target_data_path):
    """Score companies with cross-validation and multiple models"""
    # Load data
    print("Loading data...")
    company_df = pd.read_parquet(company_data_path)
    target_df = pd.read_parquet(target_data_path)
    
    # Label target companies
    company_df['is_target'] = company_df['entity_urn'].isin(target_df['entity_urn'])
    
    # Process employee highlights
    print("Processing employee highlights...")
    company_df = process_employee_highlights(company_df)
    
    # Select important features (now including derived features from employee highlights)
    numeric_features = [
        "headcount", "funding_total", "headcount_growth_12m",
        "linkedin_follower_count"
    ]
    
    # Add employee highlight features if available
    employee_features = [
        'elite_education_count', 'tech_experience_count',
        'founder_experience_count', 'team_quality_score'
    ]
    
    # Add any features that exist in the data
    for feature in employee_features:
        if feature in company_df.columns:
            numeric_features.append(feature)
    
    categorical_features = ["funding_stage"]  # Just funding stage, no country
    
    print(f"Using {len(numeric_features)} numeric features and {len(categorical_features)} categorical features")
    
    # Add derived features
    if "funding_total" in company_df.columns and "headcount" in company_df.columns:
        # Calculate funding efficiency (funding per employee)
        company_df['funding_per_employee'] = company_df['funding_total'] / company_df['headcount'].replace(0, 1)
        numeric_features.append('funding_per_employee')
    
    # Prepare data for modeling
    X = company_df[numeric_features + categorical_features].copy()
    y = company_df['is_target']
    
    # Fill missing values
    for col in numeric_features:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    
    for col in categorical_features:
        if X[col].isna().any():
            X[col] = X[col].fillna('missing')
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Preprocessing pipeline
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    # Apply preprocessing
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Apply SMOTE to create synthetic examples
    print("Applying SMOTE to balance classes...")
    try:
        smote = SMOTE(random_state=42, k_neighbors=min(5, sum(y_train)-1))
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)
        print(f"After SMOTE: {sum(y_train_balanced)} target examples, {len(y_train_balanced) - sum(y_train_balanced)} non-target examples")
    except Exception as e:
        print(f"SMOTE failed: {e}. Using original imbalanced data.")
        X_train_balanced, y_train_balanced = X_train_processed, y_train
    
    # ===== MODEL COMPARISON =====
    print("\n===== COMPARING MULTIPLE MODELS =====")
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ),
        'LogisticRegression': LogisticRegression(
            C=1.0,
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    }
    
    # Track model performance
    model_performance = {}
    
    # Evaluate each model using cross-validation
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Perform cross-validation on the entire dataset
        metrics = evaluate_model(full_pipeline, X, y, cv=5)
        model_performance[name] = metrics
        
        # Fit on the training data
        full_pipeline.fit(X_train, y_train)
        
        # Evaluate on test set
        test_auc = roc_auc_score(y_test, full_pipeline.predict_proba(X_test)[:, 1])
        print(f"Test set AUC: {test_auc:.4f}")
    
    # Find the best model based on AUC
    best_model_name = max(model_performance, key=lambda k: model_performance[k]['auc'])
    print(f"\nBest model: {best_model_name} (AUC: {model_performance[best_model_name]['auc']:.4f})")
    
    # Hyperparameter tuning for the best model
    print(f"\nPerforming hyperparameter tuning for {best_model_name}...")
    
    # Different param_grid based on best model
    if best_model_name == 'RandomForest':
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [3, 5, 7],
            'classifier__min_samples_leaf': [3, 5, 7]
        }
    elif best_model_name == 'GradientBoosting' or best_model_name == 'XGBoost':
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.1, 0.2]
        }
    elif best_model_name == 'LightGBM':
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__num_leaves': [15, 31, 63]
        }
    else:  # LogisticRegression
        param_grid = {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__penalty': ['l2', None]
        }
    
    # Create pipeline with best model
    best_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', models[best_model_name])
    ])
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        best_pipeline,
        param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Evaluate best model on test set
    best_model = grid_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Tuned model test set AUC: {test_auc:.4f}")
    
    # Score all companies using the best model
    print("Scoring all companies with best model...")
    probas = best_model.predict_proba(X)[:, 1]
    company_df['investment_score'] = (probas * 100).round(1)
    
    # Analyze scores
    target_scores = company_df[company_df['is_target']]['investment_score']
    non_target_scores = company_df[~company_df['is_target']]['investment_score']
    
    print(f"\nTarget companies: Min={target_scores.min():.1f}, Max={target_scores.max():.1f}, Mean={target_scores.mean():.2f}")
    print(f"Non-target companies: Min={non_target_scores.min():.1f}, Max={non_target_scores.max():.1f}, Mean={non_target_scores.mean():.2f}")
    
    # Calculate percentage of targets in top percentiles
    top_10_percentile = np.percentile(company_df['investment_score'], 90)
    top_25_percentile = np.percentile(company_df['investment_score'], 75)
    
    targets_in_top_10 = (target_scores >= top_10_percentile).mean() * 100
    targets_in_top_25 = (target_scores >= top_25_percentile).mean() * 100
    
    print(f"Percentage of target companies in top 10%: {targets_in_top_10:.1f}%")
    print(f"Percentage of target companies in top 25%: {targets_in_top_25:.1f}%")
    
    # Show feature importance for tree-based models
    if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
        feature_names = numeric_features.copy()
        
        # Try to get categorical feature names
        try:
            cat_features = best_model.named_steps['preprocessor'].transformers_[1][1].named_steps['encoder'].get_feature_names_out(categorical_features)
            feature_names.extend(cat_features)
        except:
            pass
        
        # Get importances
        importances = best_model.named_steps['classifier'].feature_importances_
        
        # Show top 10 features
        print("\nTop 10 Most Important Features:")
        feature_importance = {}
        for i in range(min(len(importances), len(feature_names))):
            feature_importance[feature_names[i]] = importances[i]
        
        # Sort importances
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_importance[:10]):
            print(f"{i+1}. {feature}: {importance:.4f}")
        
        # Visualize feature importance
        try:
            plt.figure(figsize=(10, 6))
            top_features = dict(sorted_importance[:10])
            plt.barh(list(top_features.keys()), list(top_features.values()))
            plt.xlabel('Importance')
            plt.title('Top 10 Feature Importance')
            plt.tight_layout()
            plt.savefig("mnt/data/feature_importance.png")
            print("Feature importance plot saved to mnt/data/feature_importance.png")
        except Exception as e:
            print(f"Error creating feature importance plot: {e}")
    
    # Generate score distribution plot
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(non_target_scores, bins=20, alpha=0.5, label='Non-Target Companies')
        plt.hist(target_scores, bins=20, alpha=0.5, label='Target Companies')
        plt.xlabel('Investment Score')
        plt.ylabel('Count')
        plt.title('Distribution of Investment Scores')
        plt.legend()
        plt.tight_layout()
        plt.savefig("mnt/data/score_distribution.png")
        print("Score distribution plot saved to mnt/data/score_distribution.png")
    except Exception as e:
        print(f"Error creating score distribution plot: {e}")
    
    # Top companies
    result_df = company_df[['entity_urn', 'name', 'investment_score', 'is_target'] + numeric_features + categorical_features].copy()
    result_df = result_df.sort_values('investment_score', ascending=False)

    # Export scored CSV
    company_df.to_parquet("mnt/data/company_data_scored.parquet", index=False)

    # Save pipeline
    joblib.dump(best_model, "mnt/data/vc_scoring_pipeline.joblib")
    
    return result_df, best_model

# Function to create model performance comparison chart
def create_model_comparison_chart(model_performance):
    """Creates a bar chart comparing model performance metrics"""
    models = list(model_performance.keys())
    auc_scores = [model_performance[model]['auc'] for model in models]
    precision_scores = [model_performance[model]['precision'] for model in models]
    recall_scores = [model_performance[model]['recall'] for model in models]
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.25
    
    plt.bar(x - width, auc_scores, width, label='AUC')
    plt.bar(x, precision_scores, width, label='Precision')
    plt.bar(x + width, recall_scores, width, label='Recall')
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("mnt/data/model_comparison.png")
    print("Model comparison chart saved to mnt/data/model_comparison.png")

# Example usage
if __name__ == "__main__":
    company_data_path = "mnt/data/company_data.parquet"
    target_data_path = "mnt/data/target_company_data.parquet"
    
    results, best_model = score_companies(company_data_path, target_data_path)
    
    print("\nTop 10 Investment Targets:")
    print(results.head(10)[['name', 'investment_score']])