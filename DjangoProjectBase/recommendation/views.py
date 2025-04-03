import numpy as np
from django.shortcuts import render
from movie.models import Movie

def recommendation(request):
    recommendation = None

    if request.method == "POST":
        prompt = request.POST.get("prompt")  # Obtén el prompt ingresado por el usuario

        # Simulación: Generar un embedding para el prompt (reemplaza con tu lógica real)
        prompt_embedding = np.random.rand(1536)

        # Buscar la película más similar al prompt
        best_match = None
        best_score = -1
        for movie in Movie.objects.all():
            movie_embedding = np.frombuffer(movie.emb, dtype=np.float32)
            similarity = np.dot(prompt_embedding, movie_embedding) / (
                np.linalg.norm(prompt_embedding) * np.linalg.norm(movie_embedding)
            )
            if similarity > best_score:
                best_match = movie
                best_score = similarity

        # Asignar la película recomendada
        recommendation = best_match

    return render(request, 'recommendation/recommendation.html', {'recommendation': recommendation})