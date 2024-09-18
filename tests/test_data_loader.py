import pytest
import torch as pt
from data_loader import load_data, apply_mask

def test_load_data_valid():
    """Test che verifica il caricamento corretto dei dati."""
    loader, times, fields, pts = load_data()
    
    assert len(times) > 0, "Times list is empty"
    assert len(fields) > 0, "Fields dictionary is empty"
    assert pts.size(0) > 0, "Tensor containing grid vertices is empty"

def test_load_data_empty():
    """Test che verifica che venga sollevato un errore se i dati sono vuoti o malformati."""
    with pytest.raises(ValueError):
        load_data(dataset_path="non/existing/path")

def test_apply_mask_valid():
    """Test che verifica l'applicazione della maschera sui punti."""
    _, _, _, pts = load_data()
    
    mask = apply_mask(pts)
    assert mask.size(0) > 0, "Mask is empty"

def test_apply_mask_empty():
    """Test che verifica che venga sollevato un errore se la maschera è vuota."""
    # Crea dei punti che sono fuori dai limiti della maschera
    pts_outside_bounds = pt.tensor([[100, 100], [200, 200]], dtype=pt.float32)
    
    with pytest.raises(ValueError):
        apply_mask(pts_outside_bounds)

def test_apply_mask_invalid_input():
    """Test che verifica che venga sollevato un errore se i punti non sono validi."""
    # Crea un input non valido (ad esempio una lista invece di un tensore)
    pts_invalid = [[0.5, 0.5], [1.0, 1.0]]  # Questo non è un tensore PyTorch

    # Verifica che venga sollevata un'eccezione RuntimeError a causa di un input non valido
    with pytest.raises(RuntimeError):
        apply_mask(pts_invalid)



