$(document).ready(function(){

	function capitalize(string) {
		return string.charAt(0).toUpperCase() + string.slice(1);
	}


	function arm_button(button, is_correct, quiz) {
		var button = $(button);
		var response_elm = quiz.find('.response');
		var prefix_elm = quiz.find('.prefix');

		var prefix;
		var class_to_add;
		var class_to_remove;

		if(is_correct) {
			prefix = 'Correct! ';
			class_to_add = 'correct';
			class_to_remove = 'incorrect';

		} else {
			prefix = 'Whoops! ';
			class_to_add = 'incorrect';
			class_to_remove = 'correct';
		}

		button.on('click', function(){
			prefix_elm.text(prefix);
			quiz.addClass(class_to_add);
			quiz.removeClass(class_to_remove);
		});
	}

	function arm_quiz(i){
		var that = $(this);

		var query = capitalize(that.find('.query-word').text());
		var correct = that.find('.correct-answer').text();
		var buttons = that.find('.option input');
		
		for(i=0; i<buttons.length; i++) {
			is_correct = $(buttons[i]).parent().hasClass(correct + '-option');
			arm_button(buttons[i], is_correct, that);
		}

		//arm_button(buttons[0], correct=='non-relational', that);
		//arm_button(buttons[1], correct=='partly-relational', that);
		//arm_button(buttons[2], correct=='relational', that)
	}


	function arm_cheat_sheet() {
	  $('.toggle-cheat').on('click', toggle_cheat);
	  $('.kill-cheat').on('click', toggle_cheat);
	}

	function toggle_cheat() {
	   var cheatsheet = $('#cheatsheet');
	   var toggler = cheatsheet.find('.toggle-cheat');
	   if (toggler.text() == '˄ Show cheatsheet') {
		  toggler.text('˅ Hide cheatsheet');
		  $('.cheatsheet').addClass('shown');
		} else {
		  toggler.text('˄ Show cheatsheet');
		  $('.cheatsheet').removeClass('shown');
		}
	}

	function arm_question(i) {
		var that = $(this);
		var options = that.find('.cml_row');
		$(options[0]).find('input').on('click', function(){
		  for (var i=1; i<options.length; i++) {
			$(options[i]).find('input').prop('checked', false);
		  }
		});
		for (var i=1; i<options.length; i++) {
		  $(options[i]).find('input').on('click', function(){
			$(options[0]).find('input').prop('checked', false);
		  })  
		}
	}


	function transplant_cheat_sheet() {
	  $('#cheatsheet').appendTo('body');
	}

	function arm_expand_example(i) {
	  $(this).find('h3').on('click', function(that){
		return function() {
		  var expandable = that.find('.expandable');
		  if(expandable.hasClass('expanded')) {
			expandable.removeClass('expanded');
			that.find('.plus-minus').prop('src', 'https://cgi.cs.mcgill.ca/~enewel3/temp/relational-nouns/plus.png');
		  } else {
			expandable.addClass('expanded');
			that.find('.plus-minus').prop('src', 'https://cgi.cs.mcgill.ca/~enewel3/temp/relational-nouns/minus.png');
		  }
		};
	  }($(this)));
	}

	function arm_show_relatum(i) {
	  $(this).find('.show-relatum').on('click', function(){
		$(this).parent().find('.refobj-target').addClass('active');
	  });
	}

	var quizes = $('.quiz');
	quizes.each(arm_quiz);

	var questions = $('.question');
	questions.each(arm_question);

	transplant_cheat_sheet();
	arm_cheat_sheet();

	$('.expand-example').each(arm_expand_example);
	$('.relatum-example').each(arm_show_relatum);

});
