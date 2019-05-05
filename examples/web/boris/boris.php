<?php

	$indir='unclassified/';

	$conn = odbc_connect("Driver={ODBC Driver 17 for SQL Server};Server=localhost;Database=LA;", 'username', 'password');
	if ( $_GET['class'] == 'undo previous' ) {
		$sqlcmd="delete from boris_classify where filename = '".$_GET['prevfilename']."'";
		odbc_exec($conn,$sqlcmd);
	} else {
		if ( $_GET['filename'] ) {
			$sqlcmd="delete from boris_classify where filename = '".$_GET['filename']."'";
			odbc_exec($conn,$sqlcmd);
			$sqlcmd="insert into boris_classify(filename,class) values('".$_GET['filename']."','".$_GET['class']."')";
			odbc_exec($conn,$sqlcmd);
		}
	}

	$previmg=$indir.$_GET['filename'];

	$sqlcmd="select top 1 filename, class, confidence from boris_classify where createdby='ml'";
	$result=odbc_exec($conn,$sqlcmd);
	$filename='none';
	$predict='';
	$confidence='';
	while ( $row = odbc_fetch_array($result) ) { 
		$filename=$row['filename'];
		$predict=$row['class'];
		$confidence=$row['confidence'];
		break;
	}

	$sqlcmd="select valid_acc from boris_validation_acc";
	$result=odbc_exec($conn,$sqlcmd);
	$valid_acc=0.0;
	while ( $row = odbc_fetch_array($result) ) { 
		$valid_acc=$row['valid_acc'];
		break;
	}

	if ( $filename == 'none' ) {
	        $filelist=scandir($indir);
       		$filename=$filelist[rand(3,sizeof($filelist))];
	}
	$result=odbc_exec($conn,$sqlcmd);

	$message='<h3>image, '.$filename.'<br />predicted to belong to the class, <span class="note">"'.$predict.'"</span>,<br />with confidence of: <span class="note">'.$confidence.'</span></h3>';

	$img=$indir.$filename;
?>
<html>
	<head>
		<style>
			.classify {position:fixed;left:600px;top:50px;font-size:24px;background-color:rgb(200,255,200);padding:10px;border-radius:20px;}
			.stats {position:fixed;left:600px;top:10px;font-size:24px;background-color:rgb(200,200,255);padding:15px;border-radius:10px;}
			.note {color:rgb(255,0,0);}
			.classify input {font-size:24px;padding:8px;border:1px black solid;}
			.classify table {font-size:32px;}
			.check {height:32px;width:32px;}
			.show {left:20px;top:10px;width:512px;border:1px black solid;}
			.showsmall {width:256px;}
			body {font-family:arial;}
		</style>
	</head>
	<body>
		<div class="show"><img src="<?php print ( $img ); ?>" class="show" /></div>
		<h3><?php print ( $message ); ?></h3>
		<hr />


		<div class="stats">holdout score: <?php print ( $valid_acc ) ?> </div>
		<form class="classify">

			<input type="hidden" name="prevfilename" value="<?php print ( $_GET['filename'] ); ?>" />
			<input type="hidden" name="filename" value="<?php print ( $filename ); ?>" />

			<a href="boris.php">reset</a>

				<?php
				        print ( '
				                        <table><tr><td>
				        ' );
				
				
			        $classes = array('Angela Merkel','Boris Johnson','Donald Trump','Malcolm Turnbull','Theresa May');
			        foreach ($classes as $class) {
			                        print ('<tr><td><input name="class" type="radio" value="'.$class.'" class="check"');
			                        if ( $class == $predict ) {
			                                print ( ' checked ' );
			                        }
			                        print ('>'.$class.'</input></td></tr>');
			        }
			        print ( '
			                        <tr><td><p>&nbsp;</p></td></tr>
			                        <tr><td><p>&nbsp;</p></td></tr>
			                        <tr><td><input name="class" type="radio" value="undo previous" class="check">undo previous</input></td></tr>
			                        <tr><td><input name="class" type="radio" value="discard" class="check">discard</input></td></tr>
			                        <tr><td><p>&nbsp;</p></td></tr>
			                        <tr><td><p>&nbsp;</p></td></tr>
			                        <tr><td><input name="go" type="submit" value="submit" /></td></tr>
			
			                        </td></tr></table>
			        ' );
			
			?>






		</form>
			<pre>
			<?php
				$sqlcmd="select count(1) as mycount from boris_classify where createdby != 'ml' and createdby != 'holdout'";
				$result=odbc_exec($conn,$sqlcmd);
				while ( $row = odbc_fetch_array($result) ) { print ( 'Count Done: ' . $row['mycount'] ); break; }
				print ( "\n" );
				$sqlcmd="select class, count(1) as mycount from boris_classify where createdby != 'ml' and createdby != 'holdout' group by class order by count(1) desc";
				$result=odbc_exec($conn,$sqlcmd);
				while ( $row = odbc_fetch_array($result) ) { print (  $row['class'] .'	:	' . $row['mycount'] . "\n" ); }
			?>
			</pre>
			<img src="<?php print ( $previmg ); ?>" class="showsmall" />
			<pre>
			<?php
				print_r ( $_GET );
			?>
	</body>
</html>
