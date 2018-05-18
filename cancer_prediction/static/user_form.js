$(document).ready(function(){
    var loading_gif = $('#loading_gif');
    $(document).on('submit','#user_form',function() {
        $("#predict_results").modal('show');
        loading_gif.show();
         var request_method = $(this).attr("method")
         var form_data = $(this).serialize()
        $.ajax({ // create an AJAX call...
            data: form_data, // get the form data
            type: request_method, // GET or POST
            url: "form_submit/", // the file to call
            success: function(response) { // on success..
                loading_gif.hide()
                $('#predict_results .modal-body').html(response); // update the DIV
            },
            error: function(e, x, r) { // on error..
                $('#predict_results .modal-body').html(e); // update the DIV
            }
        });
        return false;
    });
});