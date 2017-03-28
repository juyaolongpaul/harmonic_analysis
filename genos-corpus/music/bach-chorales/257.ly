\version "2.10.33"

\header {
  title = "257 - Nun laßt uns Gott, dem Herren"
  composer = "J. S. Bach"
}


global = {
  \time 3/4
  \key bes \major
}


soprano = \relative c'' {
  \partial 4 bes4 
  bes2 a4 
  g4 a4 bes4 
  c2. 
  bes2 bes4 
  bes2 c4   %5
  a4 g4 f4 
  bes2. 
  a2 a4 
  bes2 bes4 
  c2 d4   %10
  c2. 
  c2 d4 
  ees2 d4 
  c4 d8 c8 bes4~ 
  bes8 c8 c4. bes8   %15
  bes2
  
}

alto = \relative c' {
  \partial 4 f4 
  g2 f4 
  ees4 f4 f4 
  g4 f2 
  f2 f4 
  f2 g4   %5
  f2 f4 
  f4 e8 d8 e4 
  f2 fis4 
  g2 g4 
  ees2 d4   %10
  e4 f8 e8 g8 f8 
  f2 a4 
  bes4 c4 f,4 
  g4 a4 d,8 ees8 
  f4 g4 f4   %15
  f2
  
}

tenor = \relative c' {
  \partial 4 d4 
  d2 d8 c8 
  bes4 c4 bes4 
  bes2 a4 
  d2 d4 
  d2 c4   %5
  c2 c4 
  d4 c4 bes4 
  c2 d4 
  d2 ees4 
  a,2 bes4   %10
  bes4 a8 g8 bes8 a8 
  a2 d4 
  d4 c4 bes4 
  bes4 a4 bes4 
  bes2 a4   %15
  d2 
  
}

baixo = \relative c' {
  \partial 4 bes4 
  g2 d4 
  ees2 d4 
  ees4 f2 
  bes,2 bes'4 
  d,2 e4   %5
  f4 g4 a4 
  g2. 
  f2 d4 
  g2 ees4 
  c2 bes4   %10
  c2. 
  f2 fis4 
  g4 a4 bes4 
  e,4 fis4 g4 
  d4 ees4 f4   %15
  bes,2
  
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