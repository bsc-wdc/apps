use Bio::SearchIO;

my $blast = shift;

my $in = new Bio::SearchIO(-format=>'blast',
	-file=> $blast);

#print "db = " . $in->database_name . "\n";
$db =~ s/ //g;

while (my $res = $in->next_result){
	my $db = $res->database_name;
	$db =~ s/ //g;
	$db =~ s/\s//g;
	while(my $hit = $res->next_hit){
		my $name = $hit->name;
		if ($name =~ /(.+)\|(.+)/){
			$name = $1;
		}
		print $name . "\t" . $db . "\n";
	}
}
