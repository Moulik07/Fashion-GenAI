document.addEventListener("DOMContentLoaded", function () {
    const productsContainer = document.getElementById("products");
    const cartItemsContainer = document.getElementById("cart-items");
    const checkoutButton = document.getElementById("checkout");
    const viewCartButton = document.getElementById("view-cart");
    const cartContainer = document.getElementById("cart");
    const userForm = document.getElementById("user-form"); 
    const outfitRecommendations = document.getElementById("outfit-recommendations"); 
    // const products = [
    //   {
    //     id: 1,
    //     name: "Jeans",
    //     category: "Casual",
    //     color: "Black",
    //     brand: "Urbano Fashion",
    //     occasion: "Party",
    //     weather: "Cloudy",
    //     price: 330,
    //     image: "Flipkart Dataset Images/1.png",
    //     link:
    //       "https://www.flipkart.com/urbano-fashion-slim-men-black-jeans/p/itmf3vzvepvpz8by?pid=JEAEP6TYU7HNGRRW&lid=LSTJEAEP6TYU7HNGRRWZJCSOX&marketplace=FLIPKART&q=jeans&store=clo%2Fvua%2Fk58&srno=s_1_1&otracker=search&otracker1=search&fm=Search&iid=en_SY5cvUD9k_y7sQgM6ozpe22pa6JyVYaGXGhgVIdVWBUcWos0viBqnurV7PkASCyFC7oVx6hsJDX3Muvl10cP4w%3D%3D&ppt=sp&ppn=sp&ssid=nrr37j121c0000001692361167991&qH=a0f2589b1ced4dec"
    //   },
    //   {
    //     id: 2,
    //     name: "Jeans",
    //     category: "Trendy",
    //     color: "Blue",
    //     brand: "Brand A",
    //     occasion: "Beach",
    //     weather: "Hot",
    //     price: 450,
    //     image: "Flipkart Dataset Images/2.png"
    //   }
    // ];
    
    const cart = [];
  
    // Populate products
    products.forEach((product) => {
      const productElement = document.createElement("div");
      productElement.className = "product";
      productElement.innerHTML = `
        <h3>${product.name}</h3>
        <p>Price: $${product.price}</p>
        <button class="add-to-cart" data-id="${product.id}">Add to Cart</button>
      `;
      productsContainer.appendChild(productElement);
    });
  
    // Add to cart
    productsContainer.addEventListener("click", function (event) {
      if (event.target.classList.contains("add-to-cart")) {
        const productId = parseInt(event.target.getAttribute("data-id"), 10);
        const productToAdd = products.find((product) => product.id === productId);
        if (productToAdd) {
          cart.push(productToAdd);
          updateCart();
        }
      }
    });
  
    userForm.addEventListener("submit", function (event) {
      event.preventDefault();
  
      const userInput = document.getElementById("user-input").value;
      const occasion = document.getElementById("occasion").value;
      const weather = document.getElementById("weather").value;
      const budget = parseInt(document.getElementById("budget").value);
  
      // Send the form input to the server-side script using an AJAX request
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/get_outfit_recommendations", true);
      xhr.setRequestHeader("Content-Type", "application/json");
      
      const data = JSON.stringify({
        userInput: userInput,
        occasion: occasion,
        weather: weather,
        budget: budget
      });
  
      xhr.onreadystatechange = function () {
        if (xhr.readyState === XMLHttpRequest.DONE && xhr.status === 200) {
          const response = JSON.parse(xhr.responseText);
          displayOutfitRecommendations(response);
        }
      };
  
      xhr.send(data);
    });
  
    function displayOutfitRecommendations(recommendations) {
      outfitRecommendations.innerHTML = ""; // Clear previous recommendations
  
      if (recommendations.length === 0) {
        outfitRecommendations.textContent = "No matching outfits found.";
        return;
      }
  
      recommendations.forEach(outfit => {
        const outfitContainer = document.createElement("div");
        outfitContainer.classList.add("outfit-container");
  
        const img = document.createElement("img");
        img.src = outfit.imagePath;
        img.alt = "Outfit Image";
  
        outfitContainer.appendChild(img);
        outfitRecommendations.appendChild(outfitContainer);
      });
    }  

    // Remove from cart
    cartItemsContainer.addEventListener("click", function (event) {
      if (event.target.classList.contains("remove-from-cart")) {
        const productId = parseInt(event.target.getAttribute("data-id"), 10);
        const productIndex = cart.findIndex(
          (product) => product.id === productId
        );
        if (productIndex !== -1) {
          cart.splice(productIndex, 1);
          updateCart();
        }
      }
    });
  
    // Update cart
    function updateCart() {
      cartItemsContainer.innerHTML = "";
      cart.forEach((product) => {
        const cartItem = document.createElement("li");
        cartItem.innerHTML = `
          <span>${product.name}</span>
          <span>$${product.price}</span>
          <button class="remove-from-cart" data-id="${product.id}">Remove</button>
        `;
        cartItemsContainer.appendChild(cartItem);
      });
    }
  
    // Checkout
    checkoutButton.addEventListener("click", function () {
      alert("Thank you for your purchase!");
      cart.length = 0;
      updateCart();
    });
  
    // Toggle cart visibility
    viewCartButton.addEventListener("click", function () {
      cartContainer.classList.toggle("visible");
    });
  });
  

 