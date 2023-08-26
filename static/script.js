document.getElementById('outfitForm').addEventListener('submit', function(e) {
  e.preventDefault();
  fetch('/recommend', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify({
          user_id: e.target.user_id.value,
          user_input: e.target.user_input.value,
          occasion: e.target.occasion.value,
          weather: e.target.weather.value,
          budget: e.target.budget.value
      })
  })
  .then(response => response.json())
  .then(data => {
      let outfitsDiv = document.getElementById('outfitResults');
      outfitsDiv.innerHTML = '';
      data.forEach(outfit => {
          outfitsDiv.innerHTML += `<p>${outfit.hashtag}</p>`;
          outfitsDiv.innerHTML += `<img src="${outfit['Image Link']}" alt="Outfit Image">`;
          outfitsDiv.innerHTML += ` <div class="buy-now"><a href="${outfit['Purchase Link']}</div>">Buy Now</a>`;
      });
      document.getElementById('feedbackForm').style.display = 'block';
  });
});

document.getElementById('feedbackForm').addEventListener('submit', function(e) {
  e.preventDefault();
  fetch('/feedback', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify({
          user_id: e.target.user_id.value,
          outfit_id: e.target.outfit_id.value,
          feedback: e.target.feedback.value
      })
  })
  .then(response => {
      if (!response.ok) {
          throw new Error('Network response was not ok');
      }
      return response.json();
  })
  .then(data => {
      alert(data.message);
  })
  .catch(error => {
      console.error('There was a problem with the fetch operation:', error.message);
      alert('Error submitting feedback. Please try again.');
  });
});