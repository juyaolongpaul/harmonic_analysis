\version "2.10.33"

\header {
  title = "358 - Meine Seel erhebt den Herren"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \minor
}


soprano = \relative c'' {
  d2 f2 
  d4 d4 d4 d4 
  ees2 d2 
  c2 c2 
  bes1   %5
  d2 f2 
  c4 c4 c4 g4 
  bes2 a2 
  g1 
  d'2 f2   %10
  d4 d4 d4 d4 
  ees2 d2 
  c2 c2 
  bes1 
  d2 f2   %15
  c4 c4 c4 c4 
  c2 g4 a4 
  bes2 a2 
  g1~ 
  g1~   %20
  g1 
  g1 
  
}

alto = \relative c'' {
  g2 f2 
  f4 fis4 g4 a4 
  g4 f2 g4 
  g2 f2 
  f1   %5
  f1 
  f4 f4 ees4 g4 
  g2 fis2 
  d1 
  g2 a2   %10
  f2. fis4 
  g4 a4 bes2~ 
  bes4 bes4 bes4 a4 
  f1 
  r1   %15
  f4 g4 a4 bes4 
  c2 c,2 
  d2 d4 c4 
  b4 d4 g4 f4 
  ees1~   %20
  ees2 d4 c4 
  d1 
  
}

tenor = \relative c' {
  bes2 c2 
  d4 c4 bes4 a4 
  bes4 c2 bes4 
  bes2 a2 
  d1   %5
  bes1 
  a4 c4 g4 c4 
  d2. c4 
  bes1 
  bes2 c2   %10
  d4 c4 bes4 a4 
  bes4 c4 f,4 f'4 
  g2 f4 c4 
  d1 
  r2 f,4 g4   %15
  a4 bes4 c2~ 
  c4 d4 e4 fis4 
  g2 f4 ees4 
  d4 b4 c4 d4~ 
  d4 g,4 c2~   %20
  c2 b4 a4 
  b1 
  
}

baixo = \relative c' {
  g2 a2 
  bes4 a4 g4 fis4 
  g4 a4 bes4 g4 
  ees4 c4 f2 
  bes,1   %5
  bes'2 d,2 
  f4 a4 c4 ees,4 
  d4 c4 d2 
  g,1 
  g'2 f2   %10
  bes4 c4 d2
  d4 c4 bes4 d,4 
  ees4 c4 f2 
  bes,1 
  bes4 c4 d4 ees4   %15
  f2. g4 
  a4 bes4 c2 
  bes,4 c4 d4 ees4 
  f2 ees4 d4 
  c4 d4 ees4 f4   %20
  g1 
  g,1 
  
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