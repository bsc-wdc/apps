#!/usr/bin/perl
use strict;


my $infile = shift;

open(F,$infile);

my $str = "";
while (<F>){
	$str .= $_;
}
close(F);

my $id="";
my $ns="";
my $descr="";

my @lines = split('\n',$str);

my $header = @lines[0];
shift @lines;
my $seq = join('',@lines);

if ($header =~ /\>(\S+)\|(\S+) (.+)/){
	$id = $2;
	$ns = $1;
 	$descr = $3;
} elsif ($header =~ /\>(\S+) (.+)/){
	$id = $1;
	$descr = $2
} elsif ($header =~ /\>(.+)/){
	$id = $1;
}

my $genseq = "";

$genseq .= "<CommentedNASequence id='$id' namespace='$ns'>";
$genseq .= "<String articleName='SequenceString'>$seq</String>";
$genseq .= "<Integer articleName='Length'>".length($seq)."</Integer>";
$genseq .= "<String articleName='Description'>$descr</String>";
$genseq .= "</CommentedNASequence>";

print $genseq;
