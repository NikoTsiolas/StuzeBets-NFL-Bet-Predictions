$(document).ready(function() {
    $('#predictForm').on('submit', function(event) {
        event.preventDefault();
        $.ajax({
            url: '/predict',
            method: 'POST',
            data: $(this).serialize(),
            success: function(response) {
                $('#result').text('Prediction: ' + response.prediction);
            }
        });
    });
});
