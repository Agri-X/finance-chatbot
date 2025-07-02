// document.addEventListener('DOMContentLoaded', function () {
//     // Add a delay before simulating the click on readme-button
//     setTimeout(function () {
//         var readmeButton = document.getElementById('readme-button');
//         if (readmeButton) {
//             readmeButton.click();

//             // After the readmeButton click, wait a short moment for the new component to be rendered in the DOM.
//             // This delay might need to be adjusted based on how fast the component loads.
//             setTimeout(function() {
//                 const prefix = 'radix-:';
//                 const component = document.querySelector(`[id^="${prefix}"]`);

//                 if (component) {
//                     // Determine if it's a mobile screen based on window width
//                     // A common breakpoint for mobile is 768px, but you can adjust this as needed
//                     const isMobile = window.innerWidth <= 768;

//                     if (isMobile) {
//                         component.style.setProperty('width', '90vw', 'important');
//                     } else {
//                         component.style.setProperty('width', '70%', 'important');
//                     }

//                     component.style.setProperty('height', '90vh', 'important');
//                     // If you want to center it, you might need to set display to flex/grid on parent
//                     // or use margin: auto if the element is block-level and has a defined width.
//                     // component.style.setProperty('margin', 'auto', 'important');

//                     console.log("radix-: component found and styled!");
//                 } else {
//                     console.warn("Component with ID starting with 'radix-:' not found after click.");
//                 }
//             }, 0); // Increased delay slightly to 500ms for better reliability.
//                      // You might need to increase this if the component takes longer to appear.

//         } else {
//             console.warn("'readme-button' not found.");
//         }
//     }, 1000); // 1-second delay for initial readme-button click
// });