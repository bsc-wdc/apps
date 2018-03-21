#!/usr/bin/perl


my $ini = shift;
my $end = shift;
my $count = $end - $ini;
my $start;
my $file = shift;

#print "$ini -- $end -- $count \n\n";

#$ini=$ini - 5000;
#$end=$end + 5000;

$ini=$ini - 50;
$end=$end + 50;

#print "$ini -- $end -- $count \n\n";
my $header = 1;
my $rc = 0;
my $seq = "";

#print "$header \n";

my $finish = 0;

my $string = "";
open(F,$file);
while(<F>){
        #if ($header){
	if (/^(>\S+)/){

		#$header = 0;
		print "$_ ";
	} else {
		$string .= $_;
	}
}
close(F);

my @arr = split("\n",$string);

foreach my $line (@arr){
#print $line;
                my $l = length($line);
                if ($l + $rc < $ini){
                        $rc += $l;
                } else {
                        if ($ini>$rc && $ini<$rc+$l ) {
                                $start = $ini - $rc;
                        } else {
                                $start = 0;
                        }
                        if ($end >= $l + $rc){
#print "A $rc " . $start . " $l\n";
                                $seq .= substr($line,$start,$l);
                        } elsif ($end < $l + $rc) {
#print "B $l -- $start $end - $rc\n";
                                #$seq .= substr($line,$start,$end-$rc);
                                $seq .= substr($line,$start,$count-$rc);
#open(FF,">ooo");
#print FF $seq;
#close(FF);

                                $finish = 1;
                        }
                        $rc += $l;
                }

        if ($finish) {
#               print "fini";
                last;
        }
#print $rc . ".";
}
#print $rc . "\n";
#print $#arr;

print $seq;


