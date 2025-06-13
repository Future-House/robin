import re
import requests
from bs4 import BeautifulSoup

def extract_mesh_subheadings(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all checkbox inputs that are subheadings
    # They have names like "EntrezSystem2.PEntrez.Mesh.Mesh_ResultsPanel.Mesh_RVFull.mbsubh"
    subheading_checkboxes = soup.find_all('input', {
        'name': 'EntrezSystem2.PEntrez.Mesh.Mesh_ResultsPanel.Mesh_RVFull.mbsubh',
        'type': 'checkbox'
    })
    
    subheadings = []
    
    for checkbox in subheading_checkboxes:
        # Find the associated label
        checkbox_id = checkbox.get('id')
        if checkbox_id:
            label = soup.find('label', {'for': checkbox_id})
            if label:
                subheading_text = label.get_text().strip()
                subheadings.append(subheading_text)
    
    return sorted(subheadings)

def get_mesh_terms(term:str) -> list[str]:
    url = f"https://www.ncbi.nlm.nih.gov/mesh/?term={term.replace(' ', '+')}"
    response = requests.get(url)
    return extract_mesh_subheadings(response.text)