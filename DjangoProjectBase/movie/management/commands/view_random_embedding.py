import numpy as np
import random
from django.core.management.base import BaseCommand
from movie.models import Movie

class Command(BaseCommand):
    help = "View the embeddings of a randomly selected movie"

    def handle(self, *args, **kwargs):
        # ✅ Fetch all movies from the database
        movies = Movie.objects.all()
        if not movies.exists():
            self.stderr.write("❌ No movies found in the database.")
            return

        # ✅ Select a random movie
        movie = random.choice(movies)
        self.stdout.write(f"🎬 Selected movie: {movie.title}")

        try:
            # ✅ Retrieve and decode the embedding
            embedding_vector = np.frombuffer(movie.emb, dtype=np.float32)
            self.stdout.write(f"Embedding (first 10 values): {embedding_vector[:10]}")
        except Exception as e:
            self.stderr.write(f"❌ Failed to retrieve embedding for {movie.title}: {e}")