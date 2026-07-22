/* MaizeScan main.js */
document.addEventListener('DOMContentLoaded', function () {
  var toggle = document.getElementById('navToggle');
  var mobile = document.getElementById('navMobile');
  if (toggle && mobile) {
    toggle.addEventListener('click', function () {
      var open = mobile.classList.toggle('open');
      toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
    document.addEventListener('click', function (e) {
      if (!toggle.contains(e.target) && !mobile.contains(e.target)) {
        mobile.classList.remove('open');
        toggle.setAttribute('aria-expanded', 'false');
      }
    });
  }
});
