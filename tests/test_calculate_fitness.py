import pytest
import numpy as np
import sys, os
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import cross_val_score
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from genetic_algorithm import calculate_fitness

# Importa a função sob teste


@pytest.fixture
def sample_data():
    """Gera dados sintéticos para o teste."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test, X, y


def test_calculate_fitness_runs_without_error(sample_data):
    """Verifica se a função roda sem lançar exceções."""
    X_train, X_test, y_train, y_test, X, y = sample_data
    model = RandomForestClassifier(random_state=42)
    score = calculate_fitness(model, X_train, y_train, X_test, y_test, X, y, cv=3, scoring="accuracy")
    assert isinstance(score, float), "O retorno deve ser um float."
    assert 0.0 <= score <= 1.0, "O fitness deve estar entre 0 e 1."

def test_fitness_improves_with_more_estimators(sample_data):
    """Verifica se aumentar o número de árvores tende a melhorar o fitness médio."""
    X_train, X_test, y_train, y_test, X, y = sample_data

    model_small = RandomForestClassifier(n_estimators=5, random_state=42)
    model_large = RandomForestClassifier(n_estimators=100, random_state=42)

    score_small = calculate_fitness(model_small, X_train, y_train, X_test, y_test, X, y, cv=3, scoring="accuracy")
    score_large = calculate_fitness(model_large, X_train, y_train, X_test, y_test, X, y, cv=3, scoring="accuracy")

    assert score_large >= score_small or np.isclose(score_large, score_small, atol=0.05), \
        "Modelos maiores devem ter fitness igual ou melhor."

def test_fitness_with_different_cv_values(sample_data):
    """Verifica se a função funciona com diferentes divisões de cross-validation."""
    X_train, X_test, y_train, y_test, X, y = sample_data
    model = RandomForestClassifier(random_state=42)

    scores = []
    for cv in [2, 3, 5]:
        score = calculate_fitness(model, X_train, y_train, X_test, y_test, X, y, cv=cv, scoring="accuracy")
        scores.append(score)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    # Garante que o código funciona com múltiplos CVs sem falhar
    assert len(scores) == 3
