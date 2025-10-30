def save_results_as_txt(results_text, save_path):
    """
    Save result text to file
    
    Args:
        results_text: Text content to save
        save_path: Save path
    """
    with open(save_path, 'w') as f:
        f.write(results_text)