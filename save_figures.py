import os

class SaveFigures:
    def __init__(self, indices):
            self.indices = indices
            
    def save_figures(self, figures, file_label, folder, dpi=160, format='jpg'):

            indices = self.indices

            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f'Created directory: {folder}')


            for i, (fig, frame_idx) in enumerate(zip(figures, indices)):
                filename = os.path.join(folder, f"{file_label}_frame_{frame_idx}.{format}")
                fig.savefig(filename, bbox_inches='tight', dpi=dpi)
                print(f"Saved: {filename}")

            print(f"All {len(figures)} figures saved successfully to {folder}/")
            
            return folder