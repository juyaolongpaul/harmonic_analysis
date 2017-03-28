\version "2.10.33"

\header {
  title = "279 - Ach Gott und Herr"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key bes \major
}


soprano = \relative c'' {
  \partial 4 bes4 
  a4 g4 f4 f4 
  g4 a4 bes4 c4 
  bes4 a4 g4 a8 bes8 
  g2 f4 bes4 
  a4 bes4 c4 c4   %5
  d8 c8 bes4 c4 f,4 
  g4 a4 bes8 c8 d4 
  c4 bes4 c2 
  bes1 
  bes2.  %10
  
}

alto = \relative c' {
  \partial 4 f4 
  f4 e4 c4 c4 
  f4 fis4 g4 a4 
  g4 f4 c4 c4 
  c2 c4 d4 
  ees4 f4 f4 f4   %5
  f8 ees8 des4 c4 f4 
  f4 e4 d4. g8 
  a4 bes2 aes4~ 
  aes4 ges2 f8 ees8 
  f2.  %10
  
}

tenor = \relative c' {
  \partial 4 d4 
  c4 c8 bes8 a4 a4 
  bes4 ees4 d4 d4 
  d4 d4 e4 f4 
  f8 e16 d16 e4 a,4 f4 
  c'4 d4 a4 a4   %5
  aes4 g4 ees'4 ees4 
  d4 g4 f4 f4 
  ees4 d4 g4 f4~ 
  f2 ees2~ 
  ees4 d8 c8 d4   %10
  
}

baixo = \relative c {
  \partial 4 bes4 
  f'4 c4 f,4 ees'4 
  des4 c4 g'4 fis4 
  g4 d'8 c8 bes8 a8 g8 f8 
  c'4 c,4 f4 d4 
  c4 bes4 f'4 f4   %5
  bes,4 ees4 aes,4 a4 
  b4 cis4 d4. e8 
  fis4 g8 f8 e4 f8 ees8 
  d4 ees8 f8 ges8 aes8 bes4 
  bes,2.  %10
  
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