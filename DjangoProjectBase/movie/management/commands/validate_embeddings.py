import numpy as np
from django.core.management.base import BaseCommand
from movie.models import Movie

class Command(BaseCommand):
    help = "Validate that embeddings are stored correctly for all movies"

    def handle(self, *args, **kwargs):
        self.stdout.write("🔍 Validating embeddings for all movies...")

        for movie in Movie.objects.all():
            try:
                # Recupera el array de embeddings desde el campo BinaryField
                embedding_vector = np.frombuffer(movie.emb, dtype=np.float32)
                # Muestra el título de la película y los primeros 5 valores del embedding
                self.stdout.write(f"{movie.title}: {embedding_vector[:5]}")
            except Exception as e:
                self.stderr.write(f"❌ Failed to retrieve embedding for {movie.title}: {e}")

        self.stdout.write(self.style.SUCCESS("🎯 Finished validating embeddings"))