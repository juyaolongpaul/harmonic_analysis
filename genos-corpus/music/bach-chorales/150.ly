\version "2.10.33"

\header {
  title = "150 - Welt, ade! ich bin dein müde"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key bes \major
}


soprano = \relative c'' {
  r2 r4 <bes d>8 <bes d>
  << {f'4. f8} \\ {d4 bes} >> <c ees>4 <a c>
  <bes d>2 <a c>
  r4 <g bes>8 <g bes> << {bes8 a bes c} \\ {g4 g} >>
  << {d'4 c} \\ {f,4. f8} >> <f bes>2   % 5
  <f d'>4 <c' ees> <a f'> <bes d>
  <a c> << {bes a2} \\ {g4 g fis} >>
  << % 8
    {
      g4 <bes d>8 <bes d> <bes ees~>2
      ees4 d c4. c8
      bes1
    } 
    \\ 
    {
      g4 s g g
      f bes2 a4
      bes1
    } 
  >>
  a4 <f a> <g bes> <a c> % 11
  << {d4. d8 cis2} \\ {a4 g a2} >>
  <a f'>4 <a f'> <g e'> << {d'} \\ {d} >>
  << {e4. e8 d2} \\ {d4 cis d2} >>
  
  \bar "||"
  \time 3/2

  <g, bes>1 <ees a c>2
  <bes' d>1 <bes f'>2
  <g bes ees>1 <ees a ees'>2
  <f bes d>1.
  <f d' f>1 <f c' f>2
  << 
    {
      f'1. 
      ees2 c2. c4
      \stemUp
      bes1.
    } 
    \\ 
    {
      bes1 bes2 
      bes1 a2
      \stemNeutral
      bes1.
    } 
  >>

}

alto = \relative c' {
  f4 f f2~
  f4 f g f
  f2 f4 f8 f
  d2. bes4
  bes c d2
  f4 ees d d
  f d d2
  d g4 g
  f4. f8 f4 f
  f1
  f4 d d e
  d4. d8 e2
  f4 f g f
  e4. e8 fis2

  \time 3/2
  g1 c,2
  f1 f2
  c1 c2
  d1.
  d1 f2
  << 
    {
      f1.
      g2 f2. f4
      f1.
    } 
    \\ 
    {
      d1 d2
      c1 c2
      d1.
    } 
  >>
}

tenor = \relative c' {
  d4 d d2~
  d4 d c c
  bes2 c4 a8 a
  bes2. bes4
  bes a bes2
  bes4 g f bes
  c g a2
  bes2. bes8 bes
  c4 bes c4. c8
  d1
  d4 d bes a
  d d, a'2
  a4 c c a
  a4. a8 a2

  \time 3/2
  g1 s2
  f1 f2
  s1*6*3/2
  
}

baixo = \relative c' {
  bes4 bes bes2~
  bes4 d, ees f
  bes,2 f4 f'8 f
  g2. g4
  bes f bes,2
  bes4 c d bes
  f g d'2
  g,2. g'8 g
  a4 bes f4. f8
  bes,1
  d4 d d c
  bes4. bes8 a2
  f4 f c' d
  a4. a8 d2

  \time 3/2
  ees1 c2
  bes1 d2
  c1 c2
  bes1.
  bes'1 a2
  bes1.
  ees,2 f2. f4
  bes,1.
}

\score {
  <<
    \new StaffGroup <<
      \override StaffGroup.SystemStartBracket #'style = #'line 
      \new Staff {
        <<
          \global
          \new Voice = "soprano" { \voiceOne \soprano }
          \new Voice = "alto" { \voiceTwo \alto }
        >>
      }
      \new Staff {
        <<
          \global
          \clef "bass"
          \new Voice = "tenor" {\voiceOne \tenor }
          \new Voice = "baixo" { \voiceTwo \baixo \bar "|."}
        >>
      }
    >>
  >>
  \layout {}
  \midi {}
}